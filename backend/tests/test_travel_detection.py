from datetime import datetime, timezone

from app.models.schemas import SocialPost
from app.nlp.travel_detection import detect_travel_permalinks


def _post(text: str, permalink: str = "https://x/1") -> SocialPost:
    return SocialPost(
        id="1", platform="instagram", type="image", group="sin_etiqueta", tags=[],
        text=text, created_utc=datetime.now(timezone.utc), score=1, permalink=permalink,
    )


class TestDetectTravelPermalinks:
    def test_detects_common_travel_phrases(self):
        cases = [
            "De viaje en Roma",
            "DE VIAJE EN NUEVA YORK",
            "de vacaciones en la playa",
            "De paso por Lisboa antes de volver",
            "Unos días en Ámsterdam",
            "Visitando a mi hermana en Londres",
        ]
        for text in cases:
            posts = [_post(text)]
            assert detect_travel_permalinks(posts) == {"https://x/1"}, text

    def test_does_not_flag_ordinary_captions(self):
        posts = [_post("Cena con amigos en mi barrio"), _post("Domingo tranquilo en casa")]
        assert detect_travel_permalinks(posts) == set()

    def test_ignores_posts_without_text(self):
        posts = [_post("")]
        assert detect_travel_permalinks(posts) == set()

    def test_only_flags_matching_permalinks_among_several(self):
        posts = [
            _post("Comiendo en mi restaurante favorito", permalink="https://x/1"),
            _post("De vacaciones en Bali", permalink="https://x/2"),
        ]
        assert detect_travel_permalinks(posts) == {"https://x/2"}
