"""
Extracción de datos públicos del usuario autenticado vía Instagram Platform
API ("Business Login for Instagram").

Principio de minimización (RGPD), ACTUALIZADO: originalmente este módulo no
pedía ni descargaba imágenes/vídeos, solo metadatos textuales. Desde la
incorporación del módulo opcional de geolocalización por imagen
(app/vision/geolocation.py), se pide también `media_url` y se descarga la
imagen en memoria de forma transitoria SOLO para extraer un embedding
DINOv2; la imagen nunca se guarda en disco ni en base de datos, y se
descarta inmediatamente tras el cálculo (coherente con el diseño stateless
del resto del proyecto). Se documenta aquí como cambio consciente de
alcance para la memoria, no como descuido de la minimización original.

Normaliza cada media al modelo genérico `SocialPost` (ver
`app/models/schemas.py`), igual que `reddit_client.py` hace con sus posts y
comentarios. Mapeo concreto para Instagram:

- `group` <- primer hashtag del caption (o "sin_etiqueta" si no hay ninguno).
  Es la aproximación más parecida a "comunidad/tema" que existe en el
  contenido de Instagram.
- `score` <- `like_count + comments_count` (proxy de engagement), ya que
  Instagram no tiene un equivalente al voto neto de Reddit.
- `media_url` <- URL directa de la imagen, usada solo por el módulo de
  geolocalización (ver arriba). None si Instagram no la expone para ese
  media (p.ej. algunos vídeos).
"""
import re
from datetime import datetime

import httpx

from app.config import settings
from app.models.schemas import SocialPost, SocialProfile

IG_GRAPH_BASE = "https://graph.instagram.com"

_HASHTAG_RE = re.compile(r"#(\w+)")


class InstagramClient:
    def __init__(self, access_token: str, ig_user_id: str):
        self._access_token = access_token
        self._ig_user_id = ig_user_id

    async def fetch_profile(self) -> SocialProfile:
        async with httpx.AsyncClient(base_url=IG_GRAPH_BASE) as client:
            me = await self._get_me(client)
            media_items = await self._fetch_media(client, limit=settings.max_media)

        return SocialProfile(
            platform="instagram",
            username=me.get("username", self._ig_user_id),
            # Instagram no expone la fecha de creación de la cuenta vía esta
            # API; se deja sin rellenar (campo opcional en SocialProfile).
            account_created_utc=None,
            bio=None,
            posts=media_items,
        )

    async def _get_me(self, client: httpx.AsyncClient) -> dict:
        resp = await client.get(
            "/me",
            params={"fields": "user_id,username", "access_token": self._access_token},
        )
        resp.raise_for_status()
        return resp.json()

    async def _fetch_media(self, client: httpx.AsyncClient, limit: int) -> list[SocialPost]:
        posts: list[SocialPost] = []
        url = f"/{self._ig_user_id}/media"
        params = {
            "fields": "id,caption,timestamp,media_type,media_url,permalink,like_count,comments_count",
            "access_token": self._access_token,
            "limit": min(limit, 100),
        }

        while url and len(posts) < limit:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            body = resp.json()

            for item in body.get("data", []):
                posts.append(self._normalize(item))

            # Paginación por cursor; a partir de la segunda página, la URL
            # "next" ya trae todos los parámetros necesarios.
            next_url = body.get("paging", {}).get("next")
            if not next_url or len(posts) >= limit:
                break
            url = next_url
            params = {}  # ya van incluidos en next_url

        return posts[:limit]

    @staticmethod
    def _normalize(item: dict) -> SocialPost:
        caption = item.get("caption", "") or ""
        hashtags = [h.lower() for h in _HASHTAG_RE.findall(caption)]
        primary_hashtag = hashtags[0] if hashtags else "sin_etiqueta"

        like_count = item.get("like_count", 0) or 0
        comments_count = item.get("comments_count", 0) or 0

        return SocialPost(
            id=item["id"],
            platform="instagram",
            type=item.get("media_type", "IMAGE").lower(),
            group=primary_hashtag,
            # Todos los hashtags del caption (no solo el primero), para que
            # attribute_inference.py no pierda señal de ubicación/ocupación
            # que aparezca en hashtags secundarios.
            tags=hashtags,
            title=None,
            text=caption,
            created_utc=datetime.fromisoformat(item["timestamp"]),
            score=like_count + comments_count,
            permalink=item.get("permalink", ""),
            media_url=item.get("media_url"),
        )
