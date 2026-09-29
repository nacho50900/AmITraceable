"""Tests de los bancos temáticos de regex (app/nlp/demographic_patterns.py),
ejercitados a través de `extract_demographics` con frases realistas de bios y
publicaciones. Los positivos son variantes que antes NO se detectaban; los
negativos, falsos positivos que un banco permisivo dejaría pasar (la regex
tiene prioridad sobre la IA en `merge_findings`, así que un falso positivo
llega tal cual al estimador de k-anonimato)."""
from datetime import datetime, timezone

import pytest

from app.models.schemas import SocialPost
from app.nlp import demographic_patterns as dp
from app.nlp.demographic_extraction import extract_demographics


def _post(text: str) -> SocialPost:
    return SocialPost(
        id="1", platform="instagram", type="image", group="g", tags=[], text=text,
        created_utc=datetime.now(timezone.utc), score=1, permalink="https://x/1",
    )


POSITIVOS = [
 # (texto, campo, esperado)
 ("Terminé la carrera de Derecho el año pasado", "nivel_estudios", "superior"),
 ("Tengo un máster en marketing", "nivel_estudios", "superior"),
 ("Soy técnico superior en sistemas", "nivel_estudios", "superior"),
 ("No terminé la ESO", "nivel_estudios", "secundaria_o_inferior"),
 ("Abandoné los estudios a los 16", "nivel_estudios", "secundaria_o_inferior"),
 ("Trabajo como ingeniera en una consultora", "ocupacion", "ingeniero"),
 ("Soy abogada penalista", "ocupacion", "abogado"),
 ("Soy profe de mates en un instituto", "ocupacion", "docente"),
 ("Soy enfermera en urgencias", "ocupacion", "sanitario"),
 ("Soy programador backend", "ocupacion", "desarrollador de software"),
 ("Soy camarero en un bar", "ocupacion", "hosteleria"),
 ("Soy albañil desde hace años", "ocupacion", "construccion"),
 ("Soy camionero", "ocupacion", "transporte"),
 ("Veo un anuncio comercial en la tele", "ocupacion", None),
 ("Odio el transporte público de mi ciudad", "ocupacion", None),
 ("Estoy estudiando Medicina en la USAL", "estudios", "medicina"),
 ("Estudio ADE en la uni", "estudios", "administracion de empresas"),
 ("Estudio informática", "estudios", "ingenieria informatica"),
 ("Estudié Psicología", "estudios", "psicologia"),
 ("Me licencié en Derecho", "estudios", "derecho"),
 ("Cursando el grado de Enfermería", "estudios", "enfermeria"),
 ("Estudio Educación Primaria", "estudios", "magisterio"),
 ("Hace 3 años que no viajo", "edad", None),
 ("Mi edad es 27 y me siento joven", "edad", 27),
 ("Cumplí 31 ayer", "edad", 31),
 ("Tengo veinticuatro años", "edad", 24),
 ("Recién cumplidos los 40", "edad", 40),
 ("Yo (24M) necesito consejo", "edad", 24),
 ("Yo (24M) necesito consejo", "sexo", "hombre"),
 ("Yo 29F me acabo de mudar", "sexo", "mujer"),
 ("Soy un tío bastante tímido", "sexo", "hombre"),
 ("Soy una tía muy activa", "sexo", "mujer"),
 ("Soy padre de dos niños", "sexo", "hombre"),
 ("Soy madre soltera", "sexo", "mujer"),
 ("Soy chica", "sexo", "mujer"),
 ("Como mujer que soy, opino", "sexo", "mujer"),
 ("I'm a 25 year old guy", "sexo", "hombre"),
 ("Vivimos en Valladolid desde 2019", "municipio", "valladolid"),
 ("Resido en Madrid", "municipio", "madrid"),
 ("Me mudé a Bilbao", "municipio", "bilbao"),
 ("Vivo actualmente en Sevilla", "municipio", "sevilla"),
 ("Trabajo actualmente en Indra", "empresa", "Indra"),
 ("Trabajo en El Corte Inglés", "empresa", "El Corte Inglés"),
 ("Trabajo en Banco Santander", "empresa", "Banco Santander"),
 ("Estudio en la Uniovi", "universidad", "Oviedo"),
 ("Estudio en la UNED", "universidad", "UNED"),
 ("Soy de la UPM", "universidad", "Politécnica de Madrid"),
 ("Universidad Complutense de Madrid", "universidad", "Complutense de Madrid"),
 ("Soy ucraniana", "nacionalidad", "extranjera"),
 ("Soy italiano", "nacionalidad", "extranjera"),
 ("Nací en Marruecos", "nacionalidad", None),
 ("Soy mexicano viviendo en España", "nacionalidad", "extranjera"),
 ("Estoy en el paro", "situacion_laboral", "parado"),
 ("Sin trabajo desde marzo", "situacion_laboral", "parado"),
 ("Estoy estudiando en la universidad", "situacion_laboral", "estudiante"),
 ("Soy ama de casa", "situacion_laboral", "otro_inactivo"),
 ("Estoy trabajando en un proyecto", "situacion_laboral", None),
 ("Leo mucho por las noches", "signo_zodiacal", None),
 ("Mi signo es escorpio", "signo_zodiacal", "escorpio (23 oct - 21 nov)"),
 ("Soy testigo de Jehová", "religion", "cristianismo"),
 ("Soy evangélico", "religion", "cristianismo"),
 ("Soy hetero", "orientacion_sexual", "heterosexual"),
 ("Soy del colectivo LGTB", "orientacion_sexual", None),
 ("Hablo catalán en casa", "lengua_materna", "catalan"),
 ("Mi lengua materna es el gallego", "lengua_materna", "gallego"),
 ("Vivo con mi mujer y mis hijos", "tipo_hogar", "pareja_con_hijos"),
 ("Vivo con mi marido", "tipo_hogar", "pareja_sin_hijos"),
 ("Tengo un grado medio de FP", "nivel_estudios", "secundaria_superior"),
 ("Vivimos mi pareja y yo", "tipo_hogar", "pareja_sin_hijos"),
]


@pytest.mark.parametrize("texto,campo,esperado", POSITIVOS)
def test_detecta_variantes(texto, campo, esperado):
    assert getattr(extract_demographics([_post(texto)]), campo) == esperado


NEGATIVOS = [
    # (texto, campo que NO debe rellenarse)
    ("Mi madre es jubilada y yo voy al cine", "situacion_laboral"),
    ("Mi novio es gay y muy feliz", "orientacion_sexual"),
    ("Mi padre tiene 60 años", "edad"),
    ("Hace 15 años que vivo aquí", "edad"),
    ("Dentro de 20 años seremos otros", "edad"),
    ("Veo un anuncio comercial en la tele", "ocupacion"),
    ("Odio el transporte público", "ocupacion"),
    ("Necesito un abogado urgente", "ocupacion"),
    ("Cristiano Ronaldo marcó dos goles", "religion"),
    ("Fui a un colegio católico", "religion"),
    ("Leo mucho por las noches", "signo_zodiacal"),
    ("Vive en Sevilla mi tío", "municipio"),
    ("Estudio en la biblioteca", "estudios"),
    ("Trabajo con Python a diario", "empresa"),
    ("Trabajo en Madrid", "empresa"),
    ("Mi jefe es un pesado", "situacion_laboral"),
    ("Mi hermana es enfermera", "ocupacion"),
    ("No terminé la carrera", "nivel_estudios"),
]


@pytest.mark.parametrize("texto,campo", NEGATIVOS)
def test_no_genera_falsos_positivos(texto, campo):
    assert getattr(extract_demographics([_post(texto)]), campo) is None


class TestHelpers:
    @pytest.mark.parametrize(
        "texto,esperado",
        [("veinticuatro", 24), ("treinta y cuatro", 34), ("treinta_y_cuatro", 34), ("cuarenta", 40), ("hola", None)],
    )
    def test_numero_en_palabras(self, texto, esperado):
        assert dp.numero_en_palabras(texto) == esperado

    def test_treinta_y_cuatro_no_se_parte_por_la_y(self):
        assert dp.edad_autodeclarada("Tengo treinta y cuatro años") == 34

    def test_tercero_solo_descarta_si_va_antes(self):
        # "soy mujer" va ANTES de "mi marido": sí es del autor.
        assert dp.sexo_autodeclarado("Soy mujer casada con mi marido") == "mujer"
        # "jubilada" va DESPUÉS de "mi madre": es de la madre.
        assert dp.situacion_laboral_autodeclarada("Mi madre es jubilada") is None

    def test_sexos_contradictorios_no_se_adivinan(self):
        assert dp.sexo_autodeclarado("Soy hombre. Soy mujer.") is None

    def test_edad_y_sexo_estilo_reddit(self):
        assert dp.edad_y_sexo_reddit("Yo (24M) necesito consejo") == (24, "hombre")
        assert dp.edad_y_sexo_reddit("Yo 29F me acabo de mudar") == (29, "mujer")
        assert dp.edad_y_sexo_reddit("Mide 24 m de largo") == (None, None)

    def test_edad_fuera_de_rango_se_descarta(self):
        assert dp.edad_autodeclarada("Tengo 8 años") is None
        assert dp.edad_autodeclarada("Tengo 130 años") is None
