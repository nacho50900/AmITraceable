"""
Tests del cliente de Reddit: extracción de posts/comentarios y su
normalización al pipeline común de fingerprint/scoring/informe.

Se mockean las llamadas a oauth.reddit.com con `respx`; no se depende de
credenciales reales.
"""
import httpx
import pytest
import respx

from app.reddit_client import RedditClient

_ME_RESPONSE = {
    "name": "usuario_prueba",
    "created_utc": 1577836800,  # 2020-01-01
    "subreddit": {"public_description": "Bio de prueba"},
}

_SUBMITTED_RESPONSE = {
    "data": {
        "children": [
            {
                "data": {
                    "id": "p1",
                    "subreddit": "madrid",
                    "title": "Un post de prueba",
                    "selftext": "Contenido del post en r/madrid",
                    "created_utc": 1735725600,
                    "score": 15,
                    "permalink": "/r/madrid/comments/p1/un_post_de_prueba/",
                }
            }
        ]
    }
}

_COMMENTS_RESPONSE = {
    "data": {
        "children": [
            {
                "data": {
                    "id": "c1",
                    "subreddit": "programming",
                    "body": "Un comentario de prueba",
                    "created_utc": 1735729200,
                    "score": 4,
                    "permalink": "/r/programming/comments/xyz/c1/",
                }
            }
        ]
    }
}


@pytest.fixture
def mock_reddit_api():
    with respx.mock:
        respx.get("https://oauth.reddit.com/api/v1/me").mock(
            return_value=httpx.Response(200, json=_ME_RESPONSE),
        )
        respx.get(url__regex=r"https://oauth\.reddit\.com/user/.+/submitted").mock(
            return_value=httpx.Response(200, json=_SUBMITTED_RESPONSE),
        )
        respx.get(url__regex=r"https://oauth\.reddit\.com/user/.+/comments").mock(
            return_value=httpx.Response(200, json=_COMMENTS_RESPONSE),
        )
        yield


@pytest.mark.asyncio
async def test_fetch_profile_normalizes_reddit_posts_and_comments(mock_reddit_api):
    client = RedditClient(access_token="fake-token")
    profile = await client.fetch_profile()

    assert profile.username == "usuario_prueba"
    assert profile.platform == "reddit"
    assert profile.bio == "Bio de prueba"
    assert len(profile.posts) == 2

    post = next(p for p in profile.posts if p.type == "post")
    assert post.group == "madrid"
    assert post.score == 15
    assert post.permalink.startswith("https://reddit.com/r/madrid/")

    comment = next(p for p in profile.posts if p.type == "comment")
    assert comment.group == "programming"
    assert comment.title is None


@pytest.mark.asyncio
async def test_fetch_profile_feeds_the_common_pipeline(mock_reddit_api, patch_spacy_model):
    from app.nlp.fingerprint import build_fingerprint
    from app.nlp.attribute_inference import infer_attributes
    from app.scoring.privacy_score import compute_score

    client = RedditClient(access_token="fake-token")
    profile = await client.fetch_profile()

    fingerprint = build_fingerprint(profile.posts)
    attrs = infer_attributes(profile.posts)
    score = compute_score(profile.posts, fingerprint, attrs)

    assert 0.0 <= score.overall_score <= 100.0
    assert fingerprint.detected_language in {"es", "en"}
