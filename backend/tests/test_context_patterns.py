"""Tests del contexto (app/nlp/context_patterns.py): ubicación detallada,
viajes y ausencias, terceros, menores y relación de pareja, más la
agregación en `infer_exposure_attributes`."""
from datetime import datetime, timezone

import pytest

from app.models.schemas import SocialPost
from app.nlp.context_patterns import detectar_contexto
from app.nlp.text_signals import infer_exposure_attributes


def _post(text: str, permalink: str = "https://x/1") -> SocialPost:
    return SocialPost(
        id=permalink, platform="instagram", type="image", group="g", tags=[], text=text,
        created_utc=datetime.now(timezone.utc), score=1, permalink=permalink,
    )


POSITIVOS = [
    ("Vivo en el barrio de La Ería", "ubicacion_detallada", "barrio o la zona donde vive: La Eria"),
    ("Vivo muy cerca de la plaza mayor", "ubicacion_detallada", "vive cerca de: Plaza Mayor"),
    ("Trabajo en la zona de Gran Vía", "ubicacion_detallada", "zona donde trabaja"),
    ("Voy siempre a la cafetería del campus", "ubicacion_detallada", "lugar que frecuenta"),
    ("Comparto mi ubicación en tiempo real", "ubicacion_detallada", "tiempo real"),
    ("📍 Café Dos Ríos, Oviedo", "ubicacion_detallada", "📍"),
    ("Nos vamos de vacaciones del 10 al 20 de agosto", "viaje_futuro", "viaje o ausencia próxima"),
    ("Me voy a Lisboa el próximo viernes", "viaje_futuro", "viaje o ausencia próxima"),
    ("Tengo el vuelo mañana a las 7", "viaje_futuro", "viaje o ausencia próxima"),
    ("Dejamos la casa sola una semana", "viaje_futuro", "vivienda quedará sola"),
    ("Estaremos fuera todo el puente", "viaje_futuro", "estará fuera"),
    ("Mi hijo Pablo cumple hoy", "tercero", "su hijo/a (P.)"),
    ("Mi bebé Lucas ya gatea", "menor", "menor de su entorno (L.)"),
    ("Mi novia Laura y yo", "tercero", "su pareja (L.)"),
    ("Mi madre Carmen está mejor", "tercero", "su madre (C.)"),
    ("Mi hija tiene 5 años", "menor", "edad de un menor"),
    ("Mi hijo de 8 meses no duerme", "menor", "edad de un menor"),
    ("Mi hijo va al colegio San Fernando", "menor", "colegio o la guardería de un menor: San Fernando"),
    ("Cada mañana llevo a mi hijo al cole", "menor", "llevar o recoger"),
    ("Mi madre tiene cáncer", "tercero", "salud o el fallecimiento"),
    ("Mi abuelo ha fallecido esta semana", "tercero", "salud o el fallecimiento"),
    ("Tengo novia desde hace dos años", "relacion", "pareja actual"),
    ("Nos casamos en junio", "relacion", "cónyuge"),
    ("Estoy soltera y feliz", "relacion", "soltero/a"),
    ("Me he divorciado hace poco", "relacion", "separación o divorcio"),
    ("Nuestro aniversario fue precioso", "relacion", "aniversario"),
]


@pytest.mark.parametrize("texto,categoria,fragmento", POSITIVOS)
def test_detecta(texto, categoria, fragmento):
    hallazgos = detectar_contexto(texto)
    assert any(h.categoria == categoria and fragmento in h.valor for h in hallazgos), (
        f"{texto!r} -> {[(h.categoria, h.valor) for h in hallazgos]}"
    )


NEGATIVOS = [
    ("Ayer volvimos de Lisboa, qué viaje", "viaje_futuro"),
    ("Estuve de vacaciones en agosto", "viaje_futuro"),
    ("Mañana es lunes otra vez", "viaje_futuro"),
    ("Mi vuelo salió hace dos días", "viaje_futuro"),
    ("Su hijo Pablo cumple hoy", "menor"),
    ("El hijo de mi vecino tiene 5 años", "menor"),
    ("Mi hijo tiene 30 años", "menor"),
    ("Mi hijo va a la universidad", "menor"),
    ("Mi jefe tiene diabetes", "tercero"),
    ("Mi madre está genial", "tercero"),
    ("Hace buen día en el barrio", "ubicacion_detallada"),
    ("Su novia es de Sevilla", "relacion"),
]


@pytest.mark.parametrize("texto,categoria", NEGATIVOS)
def test_no_genera_falsos_positivos(texto, categoria):
    assert not [h for h in detectar_contexto(texto) if h.categoria == categoria]


def test_nunca_reproduce_el_nombre_de_terceros():
    for texto in ("Mi hijo Pablo cumple hoy", "Mi novia Laura y yo", "Mi madre Carmen está mejor"):
        assert all(n not in h.valor for h in detectar_contexto(texto) for n in ("Pablo", "Laura", "Carmen"))


def test_nombre_con_tilde_se_reconoce():
    assert any("(Á.)" in h.valor for h in detectar_contexto("Mi hijo Álvaro juega al fútbol"))


def test_relacion_no_toca_estado_civil():
    from app.nlp.demographic_extraction import extract_demographics

    findings = extract_demographics([_post("Estoy casada y mi marido trabaja mucho")])
    assert findings.estado_civil is None


class TestAgregacion:
    def test_junta_identificadores_y_contexto(self):
        atributos = infer_exposure_attributes([_post("Mi mail: ana@x.es. Mi hijo Pablo va al colegio Sol", "p1")])
        categorias = {a.category for a in atributos}
        assert {"contacto_publicado", "menor"} <= categorias

    def test_evidencia_y_orden(self):
        atributos = infer_exposure_attributes([_post("DNI 12345678Z", "p1"), _post("hola", "p2")])
        assert atributos[0].evidence == ["p1"]
        assert [a.confidence for a in atributos] == sorted((a.confidence for a in atributos), reverse=True)

    def test_sin_texto(self):
        assert infer_exposure_attributes([_post("")]) == []
