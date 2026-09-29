"""Tests de los bancos de estilo de vida (app/nlp/lifestyle_patterns.py) y de
su agregación en atributos inferidos (app/nlp/text_signals.py).

Los negativos importan tanto como los positivos: una mención del tema sin
gusto/práctica en primera persona ("vi un concierto en la tele", "mi hermano
es del Barça") NO debe convertirse en un atributo de la persona."""
from datetime import datetime, timezone

import pytest

from app.models.schemas import SocialPost
from app.nlp.lifestyle_patterns import detectar_estilo_de_vida
from app.nlp.text_signals import infer_lifestyle_attributes


def _post(text: str, tags: list[str] | None = None, permalink: str = "https://x/1") -> SocialPost:
    return SocialPost(
        id=permalink, platform="instagram", type="image", group="g", tags=tags or [], text=text,
        created_utc=datetime.now(timezone.utc), score=1, permalink=permalink,
    )


# (texto, categoria, fragmento esperado en el valor)
POSITIVOS = [
    ("Me encanta el cine de terror", "aficion", "Cine y series"),
    ("Me apasiona la fotografía analógica", "aficion", "Fotografía"),
    ("En mi tiempo libre juego a la PlayStation", "aficion", "Videojuegos"),
    ("Toco la guitarra desde los 12", "aficion", "Toca un instrumento"),
    ("Soy un gran fan de los videojuegos", "aficion", "Videojuegos"),
    ("Estoy leyendo un libro buenísimo", "aficion", "Lectura"),
    ("Cocino todos los domingos", "aficion", "Cocina"),
    ("Soy del Betis desde pequeño", "aficion", "Real Betis"),
    ("Hala Madrid!! campeones", "aficion", "Real Madrid"),
    ("Soy socio del Real Oviedo", "aficion", "Real Oviedo"),
    ("Tengo un perro que se llama Toby", "mascota", "Tiene perro (dice su nombre: Toby)"),
    ("Mi gata duerme conmigo", "mascota", "Tiene gato"),
    ("Me saqué el carnet ayer", "vehiculo", "carnet"),
    ("Mi coche es un Seat viejo", "vehiculo", "Seat"),
    ("Voy al trabajo en metro todos los días", "vehiculo", "transporte público"),
    ("Vivo de alquiler con mi pareja", "vivienda", "alquiler"),
    ("Firmé la hipoteca esta semana", "vivienda", "hipoteca"),
    ("Comparto piso con tres personas", "vivienda", "Comparte piso"),
    ("Vivo con mis padres todavía", "vivienda", "padres"),
    ("Hoy tengo guardia otra vez", "rutina", "guardias"),
    ("Teletrabajo tres días a la semana", "rutina", "Teletrabaja"),
    ("Trabajo los sábados en la tienda", "rutina", "fin de semana"),
    ("Fumo un paquete al día", "habito", "Fuma"),
    ("Soy vegana desde hace dos años", "habito", "Vegano"),
    ("No bebo alcohol", "habito", "alcohol"),
    ("Hablo inglés con nivel C1", "idiomas", "inglés (nivel C1)"),
    ("Estoy aprendiendo japonés", "idiomas", "japonés"),
    ("Cobro 1500 euros al mes", "ingresos", "sueldo"),
    ("Me diagnosticaron diabetes el año pasado", "salud", "condición de salud"),
    ("Voy al psicólogo cada semana", "salud", "salud mental"),
    ("Estoy embarazada de 12 semanas", "salud", "embarazo"),
    ("Voto a Podemos desde siempre", "ideologia_politica", "voto o afiliación"),
    ("Soy de izquierdas y feminista", "ideologia_politica", "ideología"),
]


@pytest.mark.parametrize("texto,categoria,fragmento", POSITIVOS)
def test_detecta(texto, categoria, fragmento):
    hallazgos = detectar_estilo_de_vida(texto)
    assert any(h.categoria == categoria and fragmento in h.valor for h in hallazgos), (
        f"{texto!r} -> {[(h.categoria, h.valor) for h in hallazgos]}"
    )


# (texto, categoria que NO debe aparecer)
NEGATIVOS = [
    ("Vi un concierto en la tele", "aficion"),
    ("La película salió ayer", "aficion"),
    ("Mi hermano es del Barça", "aficion"),
    ("El perro de mi vecino ladra", "mascota"),
    ("Tengo un perro", "salud"),
    ("Mi madre tiene diabetes", "salud"),
    ("Mi novio vota a Vox", "ideologia_politica"),
    ("Mi padre fuma mucho", "habito"),
    ("Mi hermana vive de alquiler", "vivienda"),
    ("Odio el alcohol en gel", "habito"),
    ("Hoy no hay guardia en el hospital", "rutina"),
    ("Habla inglés mi jefe", "idiomas"),
    ("Cobra 2000 euros mi marido", "ingresos"),
    ("El coche de mi vecino es un Seat", "vehiculo"),
    ("Tengo alergia al polen y a veces me pica", "ideologia_politica"),
]


@pytest.mark.parametrize("texto,categoria", NEGATIVOS)
def test_no_genera_falsos_positivos(texto, categoria):
    assert not [h for h in detectar_estilo_de_vida(texto) if h.categoria == categoria]


def test_etiquetas_son_senal_debil():
    hallazgos = detectar_estilo_de_vida("", ["foodielife", "#viajes", "random"])
    valores = {h.valor: h.confianza for h in hallazgos}
    assert set(valores) == {"Cocina y gastronomía", "Viajar"}
    assert all(c < 0.5 for c in valores.values())


def test_salud_e_ideologia_llevan_nota_art9():
    for texto in ("Me diagnosticaron diabetes", "Voto a Vox"):
        assert all("art. 9" in h.valor for h in detectar_estilo_de_vida(texto))


def test_sin_duplicados_en_un_mismo_texto():
    hallazgos = detectar_estilo_de_vida("Me encanta el cine. Me encanta el cine.")
    assert len(hallazgos) == len(set(hallazgos))


class TestAgregacion:
    def test_repeticion_sube_la_confianza_con_tope(self):
        una = infer_lifestyle_attributes([_post("Toco la guitarra", permalink="a")])
        varias = infer_lifestyle_attributes([_post("Toco la guitarra", permalink=str(i)) for i in range(30)])
        assert varias[0].confidence > una[0].confidence
        assert varias[0].confidence <= 0.9
        assert len(varias[0].evidence) == 5

    def test_evidencia_apunta_a_los_permalinks(self):
        atributos = infer_lifestyle_attributes(
            [_post("Toco la guitarra", permalink="p1"), _post("Hoy nada", permalink="p2")]
        )
        assert atributos[0].evidence == ["p1"]

    def test_orden_por_confianza_descendente(self):
        atributos = infer_lifestyle_attributes([_post("Me encanta el cine. Toco la guitarra")])
        confianzas = [a.confidence for a in atributos]
        assert confianzas == sorted(confianzas, reverse=True)

    def test_sin_texto_ni_etiquetas(self):
        assert infer_lifestyle_attributes([_post("")]) == []
        assert infer_lifestyle_attributes([]) == []
