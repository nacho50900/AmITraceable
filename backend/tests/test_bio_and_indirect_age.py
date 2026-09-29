"""Tests del banco de biografía y de la edad indirecta (año de nacimiento o
de graduación) en app/nlp/demographic_patterns.py, ejercitados también a
través de `extract_demographics` (bio como pseudo-post de tipo "bio")."""
from datetime import date, datetime, timezone

import pytest

from app.models.schemas import SocialPost
from app.nlp import demographic_patterns as dp
from app.nlp.demographic_extraction import extract_demographics

HOY = date(2026, 9, 29)


def _post(text: str, tipo: str = "image", permalink: str = "https://x/1") -> SocialPost:
    return SocialPost(
        id=permalink, platform="instagram", type=tipo, group="g", tags=[], text=text,
        created_utc=datetime.now(timezone.utc), score=1, permalink=permalink,
    )


def _bio(text: str):
    return extract_demographics([_post(text, tipo="bio", permalink="bio")], hoy=HOY)


class TestBio:
    @pytest.mark.parametrize(
        "bio,campo,esperado",
        [
            ("Católica | Madrid", "religion", "catolicismo"),
            ("Islam y paz", "religion", "islam"),
            ("Ingeniera @ Indra", "ocupacion", "ingeniero"),
            ("Ingeniera @ Indra", "empresa", "Indra"),
            ("Ing. de caminos | Oviedo", "nivel_estudios", "superior"),
            ("Madre de 2 · Enfermera", "sexo", "mujer"),
            ("Madre de 2 · Enfermera", "ocupacion", "sanitario"),
            ("Papá de Lucas y Marta", "sexo", "hombre"),
            ("23 años | Oviedo", "edad", 23),
            ("34yo | dev", "edad", 34),
            ("Ing. Informática — Uniovi", "universidad", "Oviedo"),
            ("Grado en Derecho | @uam", "estudios", "derecho"),
            ("📍 Valladolid | 21 años", "municipio", "valladolid"),
            ("Jubilado y feliz", "situacion_laboral", "jubilado"),
            ("Estudiante | gamer", "situacion_laboral", "estudiante"),
            ("Autónoma. Diseñadora", "situacion_laboral", "activo"),
        ],
    )
    def test_detecta_sin_ancla(self, bio, campo, esperado):
        assert getattr(_bio(bio), campo) == esperado

    @pytest.mark.parametrize(
        "bio,campo",
        [
            ("10 años de experiencia en ventas", "edad"),
            ("Más de 15 años ayudando a pymes", "edad"),
            ("Amante del cine y los viajes", "religion"),
            ("Madre y padre, orgullosos", "sexo"),  # ambos sexos: no se adivina
            ("Mi madre es mi heroína", "sexo"),
            ("Mi madre es enfermera", "ocupacion"),
            ("Contacto: ana@uam.es", "universidad"),  # alias corto sin '@' propio
            ("Trabajo en Madrid", "empresa"),
        ],
    )
    def test_no_genera_falsos_positivos(self, bio, campo):
        assert getattr(_bio(bio), campo) is None

    def test_las_reglas_de_bio_solo_se_aplican_al_pseudo_post_bio(self):
        # La misma frase en una publicación normal NO se acepta sin ancla.
        f = extract_demographics([_post("Católica | Madrid", tipo="image")], hoy=HOY)
        assert f.religion is None

    def test_lo_que_ya_fijo_una_publicacion_no_se_sobrescribe(self):
        f = extract_demographics(
            [_post("Ingeniera @ Indra", tipo="bio", permalink="bio"), _post("Soy abogada penalista", permalink="p1")],
            hoy=HOY,
        )
        assert f.ocupacion == "ingeniero"  # la bio va primero en la lista real (ver generator)
        f2 = extract_demographics(
            [_post("Soy abogada penalista", permalink="p1"), _post("Ingeniera @ Indra", tipo="bio", permalink="bio")],
            hoy=HOY,
        )
        assert f2.ocupacion == "abogado"

    def test_evidencia_apunta_a_la_bio(self):
        assert _bio("Católica | Madrid").evidence["religion"] == ["bio"]


class TestEdadIndirecta:
    @pytest.mark.parametrize(
        "texto,exacta",
        [
            ("Nací el 3 de mayo de 1999", 27),
            ("Fecha de nacimiento: 15/11/1995", 30),
            ("Mi cumpleaños es el 29/09/2000", 26),  # cumple hoy
            ("Nací el 30/09/2000", 25),  # cumple mañana
        ],
    )
    def test_fecha_completa_da_edad_exacta(self, texto, exacta):
        r = dp.edad_indirecta([texto], HOY)
        assert r is not None and r.exacta == exacta

    @pytest.mark.parametrize(
        "texto,rango",
        [
            ("Nací en 1999", (26, 27)),
            ("Soy del 99", (26, 27)),
            ("Quinta del 95", (30, 31)),
            ("I was born in 1990", (35, 36)),
            ("Me gradué de la carrera en 2019", (27, 34)),
            ("Terminé el máster en 2021", (26, 37)),
            ("Hice la selectividad en 2015", (27, 32)),
            ("Empecé la carrera en 2018", (24, 33)),
        ],
    )
    def test_anio_da_rango(self, texto, rango):
        r = dp.edad_indirecta([texto], HOY)
        assert r is not None and r.exacta is None and (r.minima, r.maxima) == rango

    @pytest.mark.parametrize(
        "texto",
        [
            "Mi hijo nació en 2015",
            "Mi hermana nací en 1999",  # tercero antes del verbo
            "Soy del Betis",
            "Soy del 20",  # edad fuera de rango
            "Nací en 1850",
            "Me gradué en 2035",  # futuro
            "Nací el 31/02/1999",  # fecha imposible
        ],
    )
    def test_no_deduce_edad(self, texto):
        assert dp.edad_indirecta([texto], HOY) is None

    def test_pistas_consistentes_se_intersectan(self):
        r = dp.edad_indirecta(["Nací en 1997", "Me gradué en la carrera en 2019"], HOY)
        assert (r.minima, r.maxima) == (28, 29)

    def test_pistas_contradictorias_se_descartan(self):
        assert dp.edad_indirecta(["Nací en 1999", "Terminé la carrera en 2010"], HOY) is None

    def test_rango_demasiado_ancho_se_descarta(self, monkeypatch):
        monkeypatch.setattr(dp, "_ANCHO_MAXIMO_RANGO", 5)
        assert dp.edad_indirecta(["Terminé el doctorado en 2005"], HOY) is None

    def test_fecha_completa_gana_a_cualquier_rango(self):
        r = dp.edad_indirecta(["Terminé la carrera en 2019", "Nací el 3 de mayo de 1997"], HOY)
        assert r.exacta == 29

    def test_via_extract_demographics_exacta(self):
        f = extract_demographics([_post("Nací el 3 de mayo de 1999", permalink="p1")], hoy=HOY)
        assert f.edad == 27 and f.evidence["edad"] == ["p1"] and f.source["edad"] == "texto"

    def test_via_extract_demographics_rango_con_origen_texto(self):
        f = extract_demographics([_post("Soy del 99", permalink="p1")], hoy=HOY)
        assert (f.edad_rango_min, f.edad_rango_max) == (26, 27)
        assert f.edad is None
        assert f.source["edad_rango_min"] == "texto" and f.evidence["edad_rango_min"] == ["p1"]

    def test_edad_literal_tiene_prioridad(self):
        f = extract_demographics([_post("Tengo 40 años y nací en 1999")], hoy=HOY)
        assert f.edad == 40 and f.edad_rango_min is None

    def test_k_anonimato_usa_la_nota_de_texto(self):
        from app.scoring.k_anonymity import estimate_population_narrowing

        f = extract_demographics([_post("Soy del 99")], hoy=HOY)
        paso = next(s for s in estimate_population_narrowing(f) if s.category == "edad")
        assert paso.note_code == "edad_estimada_por_fecha" and paso.source == "texto"
        assert paso.value_raw == "26-27"
