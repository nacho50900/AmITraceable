"""
Extracción de datos públicos del usuario autenticado vía API oficial de Reddit.

Principio de minimización (RGPD): solo se piden los campos necesarios para
el análisis (texto, subreddit, timestamp, score, permalink). No se descargan
imágenes, IPs, emails ni ningún dato que Reddit no expone públicamente.
"""
from datetime import datetime, timezone

import httpx

from app.config import settings
from app.models.schemas import SocialPost, SocialProfile

REDDIT_API_BASE = "https://oauth.reddit.com"


class RedditClient:
    def __init__(self, access_token: str):
        self._headers = {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": settings.reddit_user_agent,
        }

    async def fetch_profile(self) -> SocialProfile:
        async with httpx.AsyncClient(headers=self._headers, base_url=REDDIT_API_BASE) as client:
            me = await self._get_me(client)
            posts = await self._fetch_submitted(client, limit=settings.max_posts)
            comments = await self._fetch_comments(client, limit=settings.max_comments)

        return SocialProfile(
            platform="reddit",
            username=me["name"],
            account_created_utc=datetime.fromtimestamp(me["created_utc"], tz=timezone.utc),
            bio=me.get("subreddit", {}).get("public_description") or None,
            posts=posts + comments,
        )

    async def _get_me(self, client: httpx.AsyncClient) -> dict:
        resp = await client.get("/api/v1/me")
        resp.raise_for_status()
        return resp.json()

    async def _fetch_submitted(self, client: httpx.AsyncClient, limit: int) -> list[SocialPost]:
        username = (await self._get_me(client))["name"]
        resp = await client.get(
            f"/user/{username}/submitted",
            params={"limit": min(limit, 100), "sort": "new"},
        )
        resp.raise_for_status()
        items = resp.json()["data"]["children"]

        return [
            SocialPost(
                id=item["data"]["id"],
                platform="reddit",
                type="post",
                group=item["data"]["subreddit"],
                title=item["data"].get("title"),
                text=item["data"].get("selftext", "") or item["data"].get("title", ""),
                created_utc=datetime.fromtimestamp(item["data"]["created_utc"], tz=timezone.utc),
                score=item["data"].get("score", 0),
                permalink=f"https://reddit.com{item['data']['permalink']}",
            )
            for item in items
        ]

    async def _fetch_comments(self, client: httpx.AsyncClient, limit: int) -> list[SocialPost]:
        username = (await self._get_me(client))["name"]
        resp = await client.get(
            f"/user/{username}/comments",
            params={"limit": min(limit, 100), "sort": "new"},
        )
        resp.raise_for_status()
        items = resp.json()["data"]["children"]

        return [
            SocialPost(
                id=item["data"]["id"],
                platform="reddit",
                type="comment",
                group=item["data"]["subreddit"],
                title=None,
                text=item["data"].get("body", ""),
                created_utc=datetime.fromtimestamp(item["data"]["created_utc"], tz=timezone.utc),
                score=item["data"].get("score", 0),
                permalink=f"https://reddit.com{item['data']['permalink']}",
            )
            for item in items
        ]
