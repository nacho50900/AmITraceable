"""
Tests del cliente de Instagram: extracción de media y su normalización al
pipeline común de fingerprint/scoring/informe.

Se mockean las llamadas a graph.instagram.com con `respx`; no se depende de
credenciales reales.
"""
import httpx
import pytest
import respx

from app.instagram_client import InstagramClient


@pytest.fixture
def mock_instagram_api():
    with respx.mock:
        respx.get("https://graph.instagram.com/me").mock(
            return_value=httpx.Response(200, json={"user_id": "999", "username": "usuario_prueba"}),
        )
        respx.get("https://graph.instagram.com/999/media").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "1",
                            "caption": "Un día genial en #madrid con amigos!",
                            "timestamp": "2025-01-01T10:00:00+0000",
                            "media_type": "IMAGE",
                            "permalink": "https://instagram.com/p/1",
                            "like_count": 20,
                            "comments_count": 3,
                        },
                        {
                            "id": "2",
                            "caption": "Sin etiquetas aquí",
                            "timestamp": "2025-01-02T12:00:00+0000",
                            "media_type": "VIDEO",
                            "permalink": "https://instagram.com/p/2",
                            "like_count": 5,
                            "comments_count": 1,
                        },
                    ],
                    "paging": {},
                },
            ),
        )
        yield


@pytest.mark.asyncio
async def test_fetch_profile_normalizes_instagram_media(mock_instagram_api):
    client = InstagramClient(access_token="fake-token", ig_user_id="999")
    profile = await client.fetch_profile()

    assert profile.username == "usuario_prueba"
    assert len(profile.posts) == 2

    first = profile.posts[0]
    assert first.group == "madrid"  # hashtag extraído del caption
    assert first.score == 23  # like_count + comments_count
    assert first.text == "Un día genial en #madrid con amigos!"

    second = profile.posts[1]
    assert second.group == "sin_etiqueta"  # sin hashtags en el caption


@pytest.mark.asyncio
async def test_fetch_profile_feeds_the_common_pipeline(mock_instagram_api, patch_spacy_model):
    from app.nlp.fingerprint import build_fingerprint
    from app.nlp.attribute_inference import infer_attributes
    from app.scoring.privacy_score import compute_score

    client = InstagramClient(access_token="fake-token", ig_user_id="999")
    profile = await client.fetch_profile()

    fingerprint = build_fingerprint(profile.posts)
    attrs = infer_attributes(profile.posts)
    score = compute_score(profile.posts, fingerprint, attrs)

    assert 0.0 <= score.overall_score <= 100.0
    assert fingerprint.detected_language in {"es", "en"}
