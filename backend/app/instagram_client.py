"""
Extracción de datos públicos del usuario autenticado vía Instagram Platform
API ("Business Login for Instagram").

Principio de minimización (RGPD), ACTUALIZADO: originalmente este módulo no
pedía ni descargaba imágenes/vídeos, solo metadatos textuales. Desde la
incorporación del módulo opcional de geolocalización por imagen
(app/vision/geolocation.py), se piden también las URLs de imagen (incluidas
TODAS las de un carrusel, no solo la primera -- ver `children` más abajo) y
se descargan en memoria de forma transitoria SOLO para extraer un embedding
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
- `media_urls` <- URLs directas a CADA foto de la publicación, usadas solo
  por el módulo de geolocalización (ver arriba). Para una imagen suelta es
  una lista de un elemento; para un carrusel (`media_type` =
  "CAROUSEL_ALBUM"), la API de Instagram NO devuelve `media_url` útil en el
  propio item de nivel superior -- hay que pedir el campo `children` para
  obtener la URL de cada foto individual del carrusel, ver `_normalize()`.
  Los vídeos (sueltos o dentro de un carrusel) se excluyen: el modelo de
  geolocalización solo procesa imágenes. Lista vacía si no hay ninguna foto
  analizable en la publicación.
"""
import re
from datetime import datetime

import httpx

from app.config import settings
from app.models.schemas import SocialPost, SocialProfile
from app.progress import ProgressCallback, emit_progress

IG_GRAPH_BASE = "https://graph.instagram.com"

_HASHTAG_RE = re.compile(r"#(\w+)")


class InstagramClient:
    def __init__(self, access_token: str, ig_user_id: str):
        self._access_token = access_token
        self._ig_user_id = ig_user_id

    async def fetch_profile(self, progress_callback: ProgressCallback | None = None) -> SocialProfile:
        async with httpx.AsyncClient(base_url=IG_GRAPH_BASE) as client:
            me = await self._get_me(client)
            media_items = await self._fetch_media(client, limit=settings.max_media)
            await emit_progress(progress_callback, "Leyendo publicaciones...", posts_analyzed=len(media_items))

        return SocialProfile(
            platform="instagram",
            username=me.get("username", self._ig_user_id),
            # Instagram no expone la fecha de creación de la cuenta vía esta
            # API; se deja sin rellenar (campo opcional en SocialProfile).
            account_created_utc=None,
            bio=me.get("biography"),
            full_name=me.get("name"),
            avatar_url=me.get("profile_picture_url"),
            posts=media_items,
        )

    async def _get_me(self, client: httpx.AsyncClient) -> dict:
        """Pide también `name`, `biography` y `profile_picture_url`, usados
        respectivamente por la extracción de atributos con IA (ver
        app/nlp/ai_attribute_extraction.py) y por el título del dashboard
        (foto de perfil).

        OJO (pendiente de confirmar en el panel de Meta): a diferencia del
        Graph API vía Página de Facebook, donde estos campos están
        garantizados, la documentación de "Instagram API with Instagram
        Login" (`graph.instagram.com`, la que usa este cliente) no los
        listaba de forma consistente en `/me` en el momento de escribir
        esto -- puede depender del tipo de cuenta (Business vs Creator) o
        haber cambiado. Por eso se reintenta sin ellos si Meta devuelve 400
        (campo no soportado), en vez de romper el login por un campo
        opcional. TODO: verificar en el panel de la app si `name`/
        `biography`/`profile_picture_url` llegan realmente poblados para tu
        cuenta de prueba.
        """
        resp = await client.get(
            "/me",
            params={
                "fields": "user_id,username,name,biography,profile_picture_url",
                "access_token": self._access_token,
            },
        )
        if resp.status_code == 400:
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
            "fields": (
                "id,caption,timestamp,media_type,media_url,permalink,like_count,comments_count,"
                # `children` solo aplica a media_type="CAROUSEL_ALBUM": sin
                # pedirlo explícitamente, la API no devuelve las fotos
                # individuales del carrusel, solo el item "contenedor" (cuyo
                # propio `media_url` no es fiable/útil para geolocalizar).
                "children{media_url,media_type}"
            ),
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
    def _extract_media_urls(item: dict) -> list[str]:
        """Todas las URLs de foto analizables de una publicación. En un
        carrusel, la API solo expone las fotos individuales bajo el campo
        `children` (pedido explícitamente en `_fetch_media`); el propio
        item de nivel superior no trae una `media_url` que sirva para
        geolocalizar cada foto por separado. Se excluyen los vídeos (el
        modelo de geolocalización solo procesa imágenes)."""
        if item.get("media_type") == "CAROUSEL_ALBUM":
            children = (item.get("children") or {}).get("data", [])
            return [
                child["media_url"]
                for child in children
                if child.get("media_type") == "IMAGE" and child.get("media_url")
            ]

        if item.get("media_type") == "VIDEO":
            return []

        media_url = item.get("media_url")
        return [media_url] if media_url else []

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
            media_urls=InstagramClient._extract_media_urls(item),
        )
