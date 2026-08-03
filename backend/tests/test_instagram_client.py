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


class TestMediaUrlsExtraction:
    """Cobertura de InstagramClient._extract_media_urls: cuántas fotos
    analizables (media_urls) saca de cada tipo de publicación -- clave para
    que geolocation.py pueda analizar TODAS las fotos de un carrusel, no
    solo la primera."""

    @pytest.fixture
    def mock_instagram_api_with_media_types(self):
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
                                "id": "img1",
                                "caption": "Una foto suelta",
                                "timestamp": "2025-01-01T10:00:00+0000",
                                "media_type": "IMAGE",
                                "media_url": "https://cdn.fake/single.jpg",
                                "permalink": "https://instagram.com/p/img1",
                                "like_count": 1,
                                "comments_count": 0,
                            },
                            {
                                "id": "vid1",
                                "caption": "Un vídeo suelto",
                                "timestamp": "2025-01-02T10:00:00+0000",
                                "media_type": "VIDEO",
                                "media_url": "https://cdn.fake/video.mp4",
                                "permalink": "https://instagram.com/p/vid1",
                                "like_count": 1,
                                "comments_count": 0,
                            },
                            {
                                "id": "carousel1",
                                "caption": "Un carrusel con 3 fotos y 1 vídeo",
                                "timestamp": "2025-01-03T10:00:00+0000",
                                "media_type": "CAROUSEL_ALBUM",
                                # A propósito: el item "contenedor" del carrusel no
                                # trae media_url útil (o ninguna) -- las fotos
                                # reales están en `children`, no aquí.
                                "permalink": "https://instagram.com/p/carousel1",
                                "like_count": 1,
                                "comments_count": 0,
                                "children": {
                                    "data": [
                                        {"media_type": "IMAGE", "media_url": "https://cdn.fake/c1.jpg"},
                                        {"media_type": "IMAGE", "media_url": "https://cdn.fake/c2.jpg"},
                                        {"media_type": "VIDEO", "media_url": "https://cdn.fake/c3.mp4"},
                                        {"media_type": "IMAGE", "media_url": "https://cdn.fake/c4.jpg"},
                                    ]
                                },
                            },
                        ],
                        "paging": {},
                    },
                ),
            )
            yield

    @pytest.mark.asyncio
    async def test_single_image_post_has_one_media_url(self, mock_instagram_api_with_media_types):
        client = InstagramClient(access_token="fake-token", ig_user_id="999")
        profile = await client.fetch_profile()

        img_post = next(p for p in profile.posts if p.id == "img1")
        assert img_post.media_urls == ["https://cdn.fake/single.jpg"]

    @pytest.mark.asyncio
    async def test_video_post_has_no_media_urls(self, mock_instagram_api_with_media_types):
        """Un vídeo suelto no aporta ninguna foto analizable -- aunque la
        API sí traiga un `media_url` para él (apunta al vídeo, no a una
        imagen; el modelo de geolocalización no procesa vídeo)."""
        client = InstagramClient(access_token="fake-token", ig_user_id="999")
        profile = await client.fetch_profile()

        vid_post = next(p for p in profile.posts if p.id == "vid1")
        assert vid_post.media_urls == []

    @pytest.mark.asyncio
    async def test_carousel_collects_all_its_photos_excluding_videos(self, mock_instagram_api_with_media_types):
        """El caso motivador del cambio: un carrusel con 3 fotos (y 1
        vídeo en medio, que debe excluirse) debe aportar las 3 URLs, no
        solo la primera."""
        client = InstagramClient(access_token="fake-token", ig_user_id="999")
        profile = await client.fetch_profile()

        carousel_post = next(p for p in profile.posts if p.id == "carousel1")
        assert carousel_post.media_urls == [
            "https://cdn.fake/c1.jpg",
            "https://cdn.fake/c2.jpg",
            "https://cdn.fake/c4.jpg",
        ]

    @pytest.mark.asyncio
    async def test_carousel_without_children_field_degrades_to_empty_list(self):
        """Si por lo que sea la API no devuelve `children` para un
        carrusel (campo no pedido, respuesta parcial...), no debe romper --
        simplemente no hay fotos analizables de esa publicación, en vez de
        lanzar una excepción por KeyError."""
        with respx.mock:
            respx.get("https://graph.instagram.com/me").mock(
                return_value=httpx.Response(200, json={"user_id": "999", "username": "u"}),
            )
            respx.get("https://graph.instagram.com/999/media").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "carousel_sin_children",
                                "caption": "x",
                                "timestamp": "2025-01-01T10:00:00+0000",
                                "media_type": "CAROUSEL_ALBUM",
                                "permalink": "https://instagram.com/p/x",
                                "like_count": 0,
                                "comments_count": 0,
                            },
                        ],
                        "paging": {},
                    },
                ),
            )
            client = InstagramClient(access_token="fake-token", ig_user_id="999")
            profile = await client.fetch_profile()

        assert profile.posts[0].media_urls == []
