from datetime import datetime, timezone

from app.models.schemas import SocialPost
from app.nlp.demographic_extraction import extract_demographics


def _post(text: str, permalink: str = "https://x/1", i: int = 1) -> SocialPost:
    return SocialPost(
        id=str(i),
        platform="instagram",
        type="image",
        group="sin_etiqueta",
        tags=[],
        text=text,
        created_utc=datetime.now(timezone.utc),
        score=1,
        permalink=permalink,
    )


class TestAge:
    def test_detects_tengo_x_anos(self):
        findings = extract_demographics([_post("Tengo 24 años y me encanta viajar")])
        assert findings.edad == 24
        assert findings.source["edad"] == "texto"
        assert findings.evidence["edad"] == ["https://x/1"]

    def test_detects_bare_x_anos_pattern(self):
        findings = extract_demographics([_post("Con mis 30 años ya no aguanto trasnochar")])
        assert findings.edad == 30

    def test_discards_out_of_range_false_positive(self):
        findings = extract_demographics([_post("Este edificio tiene 100 años de historia")])
        # 100 es el límite superior inclusive, así que no se descarta aquí;
        # probamos con un valor claramente fuera de rango humano.
        findings2 = extract_demographics([_post("Este puente tiene 200 años de historia")])
        assert findings2.edad is None

    def test_first_match_wins_does_not_overwrite(self):
        posts = [_post("Tengo 24 años", i=1), _post("Tengo 50 años", permalink="https://x/2", i=2)]
        findings = extract_demographics(posts)
        assert findings.edad == 24


class TestSex:
    def test_detects_soy_una_chica(self):
        findings = extract_demographics([_post("Soy una chica de ciudad")])
        assert findings.sexo == "mujer"

    def test_detects_soy_mujer(self):
        findings = extract_demographics([_post("Soy mujer y estoy orgullosa")])
        assert findings.sexo == "mujer"

    def test_detects_soy_un_chico(self):
        findings = extract_demographics([_post("Soy un chico normal")])
        assert findings.sexo == "hombre"

    def test_detects_soy_hombre(self):
        findings = extract_demographics([_post("Soy hombre de pocas palabras")])
        assert findings.sexo == "hombre"

    def test_no_match_leaves_none(self):
        findings = extract_demographics([_post("Hoy hace un día estupendo")])
        assert findings.sexo is None


class TestLocation:
    def test_detects_municipio_with_accents(self):
        findings = extract_demographics([_post("Vivo en León desde hace años")])
        assert findings.municipio == "leon"
        assert findings.provincia is None  # municipio tiene prioridad, no rellena los dos

    def test_detects_provincia_when_no_municipio_match(self):
        # "cuenca" está en PROVINCE_POPULATION pero no en MUNICIPALITY_POPULATION
        findings = extract_demographics([_post("Vivo en Cuenca, un sitio muy tranquilo")])
        assert findings.provincia == "cuenca"
        assert findings.municipio is None

    def test_no_vivo_en_phrase_leaves_both_none(self):
        findings = extract_demographics([_post("Me encanta viajar por España")])
        assert findings.municipio is None
        assert findings.provincia is None

    def test_unrecognized_place_name_leaves_both_none(self):
        findings = extract_demographics([_post("Vivo en un pueblo que no existe en ninguna tabla")])
        assert findings.municipio is None
        assert findings.provincia is None


class TestStudies:
    def test_detects_estudio_x(self):
        findings = extract_demographics([_post("Estudio Medicina en la universidad")])
        assert findings.estudios == "medicina"

    def test_detects_estudiante_de_x(self):
        findings = extract_demographics([_post("Soy estudiante de Enfermeria este año")])
        assert findings.estudios == "enfermeria"

    def test_detects_graduado_en_x(self):
        findings = extract_demographics([_post("Graduado en Derecho el año pasado")])
        assert findings.estudios == "derecho"

    def test_unmatched_study_field_leaves_none(self):
        findings = extract_demographics([_post("Estudio jardinería avanzada")])
        assert findings.estudios is None


class TestOccupation:
    def test_detects_known_occupation_keyword(self):
        findings = extract_demographics([_post("Trabajo como docente en un instituto")])
        assert findings.ocupacion == "docente"

    def test_no_match_leaves_none(self):
        findings = extract_demographics([_post("Hoy fui al parque con mi perro")])
        assert findings.ocupacion is None


class TestUniversity:
    def test_detects_universidad_de_x(self):
        findings = extract_demographics([_post("Estudié en la Universidad de Salamanca")])
        assert findings.universidad == "Salamanca"

    def test_no_match_leaves_none(self):
        findings = extract_demographics([_post("No menciono ninguna universidad aquí")])
        assert findings.universidad is None


class TestCompany:
    def test_detects_trabajo_en_x_lowercase(self):
        findings = extract_demographics([_post("trabajo en Indra desde hace dos años")])
        assert findings.empresa == "Indra"

    def test_detects_trabajo_capitalized_at_sentence_start(self):
        # Antes era una limitación conocida (_COMPANY_RE no usaba
        # re.IGNORECASE en "trabajo"), arreglado con una bandera inline
        # (?i:trabajo) que solo afecta a esa palabra, sin relajar el
        # requisito de mayúscula inicial en el nombre de la empresa.
        findings = extract_demographics([_post("Trabajo en Indra desde hace dos años")])
        assert findings.empresa == "Indra"

    def test_lowercase_company_name_not_matched(self):
        # La regex sigue exigiendo mayúscula inicial en el NOMBRE DE LA
        # EMPRESA (parte no afectada por la bandera case-insensitive) para
        # evitar falsos positivos con "trabajo en casa", "trabajo en remoto".
        findings = extract_demographics([_post("trabajo en remoto casi siempre")])
        assert findings.empresa is None

    def test_lowercase_company_name_not_matched_capitalized_sentence(self):
        findings = extract_demographics([_post("Trabajo en remoto casi siempre")])
        assert findings.empresa is None


class TestMultiplePostsAndEmptyInput:
    def test_empty_post_list_returns_empty_findings(self):
        findings = extract_demographics([])
        assert findings.sexo is None
        assert findings.source == {}

    def test_post_with_empty_text_is_skipped_without_error(self):
        findings = extract_demographics([_post("")])
        assert findings.sexo is None

    def test_aggregates_different_attributes_from_different_posts(self):
        posts = [
            _post("Soy una chica de ciudad", permalink="https://x/1", i=1),
            _post("Tengo 24 años y vivo en León", permalink="https://x/2", i=2),
            _post("Estudio Medicina y trabajo en Roche", permalink="https://x/3", i=3),
        ]

        findings = extract_demographics(posts)

        assert findings.sexo == "mujer"
        assert findings.edad == 24
        assert findings.municipio == "leon"
        assert findings.estudios == "medicina"
        assert findings.empresa == "Roche"
        # Cada atributo mantiene su propia evidencia del post que lo generó
        assert findings.evidence["sexo"] == ["https://x/1"]
        assert findings.evidence["edad"] == ["https://x/2"]
        assert findings.evidence["empresa"] == ["https://x/3"]

    def test_all_detected_fields_marked_as_texto_source(self):
        findings = extract_demographics([_post("Soy una chica de 24 años, vivo en León")])
        for field_name in ("sexo", "edad", "municipio"):
            assert findings.source[field_name] == "texto"
        # Los campos NO detectados no aparecen en source en absoluto
        assert "estudios" not in findings.source
