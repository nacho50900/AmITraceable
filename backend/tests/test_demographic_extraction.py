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

    def test_detects_multi_province_comunidad_autonoma(self):
        """'Canarias' no es una provincia del INE (son dos: Las Palmas y
        Santa Cruz de Tenerife), así que debe quedar a nivel de comunidad
        autónoma, no perderse."""
        findings = extract_demographics([_post("Vivo en Canarias, cerca de la playa")])
        assert findings.comunidad_autonoma == "canarias"
        assert findings.provincia is None
        assert findings.municipio is None
        assert findings.source["comunidad_autonoma"] == "texto"
        assert findings.evidence["comunidad_autonoma"] == ["https://x/1"]

    def test_detects_english_comunidad_autonoma_name(self):
        findings = extract_demographics([_post("Vivo en Andalusia toda mi vida")])
        assert findings.comunidad_autonoma == "andalucia"

    def test_single_province_comunidad_autonoma_resolves_directly_to_province(self):
        """Asturias es una comunidad de una sola provincia: como su nombre
        ya coincide con una clave de PROVINCE_POPULATION, se resuelve
        directamente como provincia (más específico), sin pasar por
        comunidad_autonoma."""
        findings = extract_demographics([_post("Vivo en Asturias, junto al mar")])
        assert findings.provincia == "asturias"
        assert findings.comunidad_autonoma is None

    def test_comunidad_autonoma_does_not_override_already_found_provincia(self):
        posts = [
            _post("Vivo en León desde hace años", i=1),
            _post("Vivo en Canarias", permalink="https://x/2", i=2),
        ]
        findings = extract_demographics(posts)
        assert findings.municipio == "leon"
        assert findings.comunidad_autonoma is None


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


class TestNationality:
    def test_detects_nacionalidad_espanola_explicita(self):
        findings = extract_demographics([_post("Soy español y vivo fuera desde hace años")])
        assert findings.nacionalidad == "espanola"
        assert findings.source["nacionalidad"] == "texto"

    def test_detects_nacionalidad_extranjera_por_gentilicio(self):
        findings = extract_demographics([_post("Soy colombiana viviendo en Madrid")])
        assert findings.nacionalidad == "extranjera"

    def test_no_nationality_mention_stays_none(self):
        findings = extract_demographics([_post("Me encanta el fútbol")])
        assert findings.nacionalidad is None


class TestEmploymentStatus:
    def test_detects_parado(self):
        findings = extract_demographics([_post("Estoy en paro desde marzo, buscando oportunidades")])
        assert findings.situacion_laboral == "parado"

    def test_detects_jubilado(self):
        findings = extract_demographics([_post("Jubilado y disfrutando de la huerta")])
        assert findings.situacion_laboral == "jubilado"

    def test_detects_estudiante(self):
        findings = extract_demographics([_post("Soy estudiante y esto me quita mucho tiempo")])
        assert findings.situacion_laboral == "estudiante"

    def test_detects_activo(self):
        findings = extract_demographics([_post("Trabajo en un hospital desde hace 3 años")])
        assert findings.situacion_laboral == "activo"

    def test_parado_takes_priority_over_generic_work_mention_in_same_text(self):
        # "buscando empleo" (parado) no debería confundirse con "activo" aunque
        # el texto también mencione la palabra "trabajo" de forma genérica.
        findings = extract_demographics([_post("Busco trabajo desde hace meses, cualquier cosa vale")])
        assert findings.situacion_laboral == "parado"


class TestHouseholdType:
    def test_detects_unipersonal(self):
        findings = extract_demographics([_post("Vivo solo desde que me mudé a la ciudad")])
        assert findings.tipo_hogar == "unipersonal"

    def test_detects_pareja_sin_hijos(self):
        findings = extract_demographics([_post("Vivo con mi pareja en un piso pequeño")])
        assert findings.tipo_hogar == "pareja_sin_hijos"

    def test_detects_pareja_con_hijos_combining_two_posts(self):
        posts = [
            _post("Vivo con mi pareja desde hace 5 años", permalink="https://x/1", i=1),
            _post("Mis hijos ya están en el colegio", permalink="https://x/2", i=2),
        ]
        findings = extract_demographics(posts)
        assert findings.tipo_hogar == "pareja_con_hijos"
        # La evidencia combina los permalinks de ambas señales, sin duplicar
        assert set(findings.evidence["tipo_hogar"]) == {"https://x/1", "https://x/2"}

    def test_detects_monoparental_explicit_mention(self):
        findings = extract_demographics([_post("Somos una familia monoparental y muy felices")])
        assert findings.tipo_hogar == "monoparental"

    def test_detects_monoparental_from_alone_plus_children(self):
        posts = [
            _post("Vivo sola con mis hijos", permalink="https://x/1", i=1),
        ]
        findings = extract_demographics(posts)
        assert findings.tipo_hogar == "monoparental"

    def test_no_household_signal_stays_none(self):
        findings = extract_demographics([_post("Me encanta el senderismo los fines de semana")])
        assert findings.tipo_hogar is None


class TestMotherTongue:
    def test_detects_catalan(self):
        findings = extract_demographics([_post("Mi lengua materna es el catalán")])
        assert findings.lengua_materna == "catalan"

    def test_detects_euskera(self):
        findings = extract_demographics([_post("Hablo euskera con mi familia")])
        assert findings.lengua_materna == "euskera"

    def test_detects_gallego(self):
        findings = extract_demographics([_post("Soy galegofalante de nacimiento")])
        assert findings.lengua_materna == "gallego"

    def test_no_language_mention_stays_none(self):
        findings = extract_demographics([_post("Me encanta cocinar los domingos")])
        assert findings.lengua_materna is None


class TestSymbolicAttributes:
    def test_detects_zodiac_emoji_and_religion_symbol(self):
        text = "✡️🙏🙏 siempre con fé ♈ soy 4/9/09 heterosexual"
        findings = extract_demographics([_post(text)])
        assert findings.religion == "judaismo"
        assert findings.signo_zodiacal == "aries (21 mar - 19 abr)"
        assert findings.orientacion_sexual == "heterosexual"
        assert findings.source["religion"] == "texto"
        assert findings.source["signo_zodiacal"] == "texto"
        assert findings.source["orientacion_sexual"] == "texto"

    def test_detects_common_sexuality_typos(self):
        findings = extract_demographics([_post("Soy heterosexuial y me gusta la vida")])
        assert findings.orientacion_sexual == "heterosexual"

    def test_detects_zodiac_name_and_religious_self_id(self):
        findings = extract_demographics([_post("Soy aries y soy judia")])
        assert findings.signo_zodiacal == "aries (21 mar - 19 abr)"
        assert findings.religion == "judaismo"

    def test_detects_accented_and_common_religious_and_zodiac_variants(self):
        findings = extract_demographics([_post("Soy judía, soy cáncer y heterosexuial")])
        assert findings.religion == "judaismo"
        assert findings.signo_zodiacal == "cancer (21 jun - 22 jul)"
        assert findings.orientacion_sexual == "heterosexual"


class TestSportPractice:
    """Práctica deportiva: requiere un verbo de PRÁCTICA explícita, no una
    simple mención del deporte -- ver _SPORT_PRACTICE_RE en
    demographic_extraction.py. Cada caso positivo tiene un caso "gemelo"
    de mera mención que debe quedarse en None, para dejar constancia
    explícita de que se comprobó el falso positivo, no solo el caso feliz."""

    def test_detects_futbol_practice(self):
        findings = extract_demographics([_post("Juego al futbol todos los sabados con mis amigos")])
        assert findings.practica_deportiva == "futbol"

    def test_futbol_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi el partido de futbol ayer, menudo partidazo")])
        assert findings.practica_deportiva is None

    def test_detects_running_practice(self):
        findings = extract_demographics([_post("Salgo a correr cada semana antes de currar")])
        assert findings.practica_deportiva == "running"

    def test_running_generic_mention_is_not_detected(self):
        findings = extract_demographics([_post("Me gustaria correr una maraton algun dia")])
        assert findings.practica_deportiva is None

    def test_detects_natacion_practice(self):
        findings = extract_demographics([_post("Practico natacion desde que era niño")])
        assert findings.practica_deportiva == "natacion"

    def test_detects_senderismo_practice(self):
        findings = extract_demographics([_post("Este finde hago senderismo por la sierra")])
        assert findings.practica_deportiva == "senderismo"

    def test_detects_musculacion_practice(self):
        findings = extract_demographics([_post("Voy al gimnasio cuatro veces por semana")])
        assert findings.practica_deportiva == "musculacion"

    def test_detects_ciclismo_practice(self):
        findings = extract_demographics([_post("Salgo en bici todos los fines de semana")])
        assert findings.practica_deportiva == "ciclismo"

    def test_ciclismo_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi el tour de francia por la tele")])
        assert findings.practica_deportiva is None

    def test_detects_padel_practice(self):
        findings = extract_demographics([_post("Juego al padel todos los martes con mi cuñado")])
        assert findings.practica_deportiva == "padel"

    def test_padel_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Me gusta ver padel en la tele los fines de semana")])
        assert findings.practica_deportiva is None

    def test_detects_tenis_practice(self):
        findings = extract_demographics([_post("Juego al tenis con mi hermano cada semana")])
        assert findings.practica_deportiva == "tenis"

    def test_tenis_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi la final de tenis ayer, que partidazo")])
        assert findings.practica_deportiva is None

    def test_detects_baloncesto_practice(self):
        findings = extract_demographics([_post("Juego al baloncesto en el equipo del barrio")])
        assert findings.practica_deportiva == "baloncesto"

    def test_baloncesto_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi el partido de baloncesto anoche")])
        assert findings.practica_deportiva is None

    def test_detects_futbol_sala_practice(self):
        findings = extract_demographics([_post("Juego al futbol sala en una liga amateur los jueves")])
        assert findings.practica_deportiva == "futbol_sala"

    def test_detects_futbito_as_futbol_sala(self):
        findings = extract_demographics([_post("Juego al futbito con mis amigos los viernes")])
        assert findings.practica_deportiva == "futbol_sala"

    def test_futbol_sala_is_not_confused_with_futbol(self):
        """Caso motivador: 'futbol' es substring de 'futbol sala', así que
        el orden de alternancia en _SPORT_PRACTICE_RE importa -- si el
        grupo 'futbol' se comprobara primero, 'juego al futbol sala'
        haría match como 'futbol' (con \\b justo antes de 'sala'), nunca
        llegando a probar la alternativa 'futbol_sala'."""
        findings = extract_demographics([_post("Juego al futbol sala todos los jueves")])
        assert findings.practica_deportiva == "futbol_sala"
        assert findings.practica_deportiva != "futbol"

    def test_futbol_sala_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi un torneo de futbol sala este finde")])
        assert findings.practica_deportiva is None

    def test_detects_golf_practice(self):
        findings = extract_demographics([_post("Juego al golf todos los fines de semana")])
        assert findings.practica_deportiva == "golf"

    def test_golf_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi el masters de golf en la television")])
        assert findings.practica_deportiva is None

    def test_detects_yoga_practice(self):
        findings = extract_demographics([_post("Practico yoga cada manana antes de trabajar")])
        assert findings.practica_deportiva == "yoga_pilates"

    def test_detects_pilates_as_yoga_pilates(self):
        findings = extract_demographics([_post("Voy a clases de pilates dos veces por semana")])
        assert findings.practica_deportiva == "yoga_pilates"

    def test_yoga_generic_mention_is_not_detected(self):
        findings = extract_demographics([_post("Me encanta el yoga como filosofia de vida")])
        assert findings.practica_deportiva is None

    def test_detects_spinning_as_gimnasia_intensa(self):
        findings = extract_demographics([_post("Voy a spinning tres veces por semana")])
        assert findings.practica_deportiva == "gimnasia_intensa"

    def test_gimnasia_intensa_generic_mention_is_not_detected(self):
        findings = extract_demographics([_post("La gimnasia intensa quema muchas calorias")])
        assert findings.practica_deportiva is None

    def test_crossfit_stays_musculacion_not_gimnasia_intensa(self):
        """'hago crossfit' debe seguir cayendo en 'musculacion' (donde ya
        estaba antes de añadir 'gimnasia_intensa'), no detectarse dos
        veces ni cambiar de categoria."""
        findings = extract_demographics([_post("Hago crossfit en el box de mi barrio")])
        assert findings.practica_deportiva == "musculacion"

    # --- Ampliación a la tabla completa de la Encuesta de Hábitos
    # Deportivos en España 2024/25 (tabla 1.21, 41 modalidades) ---

    def test_detects_atletismo_practice(self):
        findings = extract_demographics([_post("Practico atletismo desde los 12 anos")])
        assert findings.practica_deportiva == "atletismo"

    def test_atletismo_and_running_are_distinct(self):
        """Regresión: en un borrador anterior 'practico atletismo' caia en
        el grupo 'running' -- la encuesta 2024/25 los trata como dos filas
        separadas (poblaciones muy distintas), asi que deben seguir
        siendo categorias independientes en el regex."""
        atletismo = extract_demographics([_post("Practico atletismo desde los 12 anos")])
        running = extract_demographics([_post("Salgo a correr todas las semanas")])
        assert atletismo.practica_deportiva == "atletismo"
        assert running.practica_deportiva == "running"

    def test_detects_tenis_mesa_practice(self):
        findings = extract_demographics([_post("Juego al tenis de mesa los domingos")])
        assert findings.practica_deportiva == "tenis_mesa"

    def test_detects_ping_pong_as_tenis_mesa(self):
        findings = extract_demographics([_post("Juego al ping pong con mis companeros de piso")])
        assert findings.practica_deportiva == "tenis_mesa"

    def test_tenis_mesa_is_not_confused_with_tenis(self):
        """Regresión: 'tenis' es prefijo de 'tenis de mesa' -- si el grupo
        'tenis' se comprobara primero, 'juego al tenis de mesa' haria
        match como 'tenis' antes de llegar a probar 'tenis_mesa'."""
        findings = extract_demographics([_post("Juego al tenis de mesa todos los domingos")])
        assert findings.practica_deportiva == "tenis_mesa"
        assert findings.practica_deportiva != "tenis"

    def test_detects_esqui_practice(self):
        findings = extract_demographics([_post("Voy a esquiar todos los inviernos")])
        assert findings.practica_deportiva == "esqui"

    def test_detects_esqui_nautico_practice(self):
        findings = extract_demographics([_post("Practico esqui nautico en verano")])
        assert findings.practica_deportiva == "esqui_nautico"

    def test_esqui_nautico_is_not_confused_with_esqui(self):
        """Regresión: mismo motivo que futbol/futbol_sala y tenis/tenis_mesa
        -- 'esqui' es prefijo de 'esqui nautico'."""
        findings = extract_demographics([_post("Hago esqui nautico en la costa")])
        assert findings.practica_deportiva == "esqui_nautico"
        assert findings.practica_deportiva != "esqui"

    def test_esqui_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Me gusta el esqui como deporte de invierno")])
        assert findings.practica_deportiva is None

    def test_detects_voleibol_practice(self):
        findings = extract_demographics([_post("Juego al voleibol en el equipo del instituto")])
        assert findings.practica_deportiva == "voleibol"

    def test_detects_voley_as_voleibol(self):
        findings = extract_demographics([_post("Juego al voley todos los sabados")])
        assert findings.practica_deportiva == "voleibol"

    def test_voleibol_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Me encanta ver voleibol playa en verano")])
        assert findings.practica_deportiva is None

    def test_detects_balonmano_practice(self):
        findings = extract_demographics([_post("Juego al balonmano en el equipo del barrio")])
        assert findings.practica_deportiva == "balonmano"

    def test_balonmano_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi el partido de balonmano en la tele")])
        assert findings.practica_deportiva is None

    def test_detects_rugby_practice(self):
        findings = extract_demographics([_post("Practico rugby los fines de semana")])
        assert findings.practica_deportiva == "rugby"

    def test_detects_pelota_vasca_practice(self):
        findings = extract_demographics([_post("Juego al fronton con mi padre")])
        assert findings.practica_deportiva == "pelota_vasca"

    def test_detects_frontenis_as_pelota_vasca(self):
        findings = extract_demographics([_post("Practico frontenis desde hace anos")])
        assert findings.practica_deportiva == "pelota_vasca"

    def test_detects_petanca_practice(self):
        findings = extract_demographics([_post("Juego a la petanca los domingos")])
        assert findings.practica_deportiva == "petanca"

    def test_detects_patinaje_practice(self):
        findings = extract_demographics([_post("Hago patinaje artistico desde nina")])
        assert findings.practica_deportiva == "patinaje"

    def test_detects_motociclismo_practice(self):
        findings = extract_demographics([_post("Practico motociclismo de competicion")])
        assert findings.practica_deportiva == "motociclismo"

    def test_motociclismo_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi la carrera de motoGP este finde")])
        assert findings.practica_deportiva is None

    def test_detects_automovilismo_practice(self):
        findings = extract_demographics([_post("Hago rallies desde hace anos")])
        assert findings.practica_deportiva == "automovilismo"

    def test_automovilismo_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi la formula 1 el domingo pasado")])
        assert findings.practica_deportiva is None

    def test_detects_aeronautica_practice(self):
        findings = extract_demographics([_post("Practico parapente todos los findes que puedo")])
        assert findings.practica_deportiva == "aeronautica"

    def test_detects_squash_practice(self):
        findings = extract_demographics([_post("Juego al squash con un amigo del trabajo")])
        assert findings.practica_deportiva == "squash"

    def test_detects_badminton_practice(self):
        findings = extract_demographics([_post("Juego al badminton en el pabellon")])
        assert findings.practica_deportiva == "badminton"

    def test_detects_surf_practice(self):
        findings = extract_demographics([_post("Hago surf todos los veranos en la playa")])
        assert findings.practica_deportiva == "surf"

    def test_detects_vela_practice(self):
        findings = extract_demographics([_post("Practico vela en el club nautico")])
        assert findings.practica_deportiva == "vela"

    def test_detects_piraguismo_practice(self):
        findings = extract_demographics([_post("Hago piraguismo por el rio los fines de semana")])
        assert findings.practica_deportiva == "piraguismo_remo"

    def test_detects_kayak_as_piraguismo_remo(self):
        findings = extract_demographics([_post("Practico kayak en el pantano")])
        assert findings.practica_deportiva == "piraguismo_remo"

    def test_detects_submarinismo_practice(self):
        findings = extract_demographics([_post("Practico submarinismo en Baleares cada verano")])
        assert findings.practica_deportiva == "submarinismo"

    def test_detects_buceo_as_submarinismo(self):
        findings = extract_demographics([_post("Hago buceo desde que me saque el titulo")])
        assert findings.practica_deportiva == "submarinismo"

    def test_submarinismo_generic_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi un documental de submarinismo anoche")])
        assert findings.practica_deportiva is None

    def test_detects_triatlon_practice(self):
        findings = extract_demographics([_post("Hago triatlon desde hace dos anos")])
        assert findings.practica_deportiva == "triatlon"

    def test_detects_boxeo_practice(self):
        findings = extract_demographics([_post("Practico boxeo tres veces por semana")])
        assert findings.practica_deportiva == "boxeo"

    def test_boxeo_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi la pelea de boxeo de anoche")])
        assert findings.practica_deportiva is None

    def test_detects_karate_as_artes_marciales(self):
        findings = extract_demographics([_post("Hago karate desde que era nino")])
        assert findings.practica_deportiva == "artes_marciales"

    def test_detects_judo_as_artes_marciales(self):
        findings = extract_demographics([_post("Practico judo en un club cerca de casa")])
        assert findings.practica_deportiva == "artes_marciales"

    def test_detects_defensa_personal_as_lucha_defensa_personal(self):
        findings = extract_demographics([_post("Practico defensa personal los martes")])
        assert findings.practica_deportiva == "lucha_defensa_personal"

    def test_artes_marciales_and_lucha_defensa_personal_are_distinct(self):
        artes = extract_demographics([_post("Hago karate desde que era nino")])
        lucha = extract_demographics([_post("Practico defensa personal los martes")])
        assert artes.practica_deportiva == "artes_marciales"
        assert lucha.practica_deportiva == "lucha_defensa_personal"

    def test_detects_caza_practice(self):
        findings = extract_demographics([_post("Voy de caza los fines de semana con mi padre")])
        assert findings.practica_deportiva == "caza"

    def test_detects_pesca_practice(self):
        findings = extract_demographics([_post("Salgo a pescar cada domingo por la manana")])
        assert findings.practica_deportiva == "pesca"

    def test_detects_hipica_practice(self):
        findings = extract_demographics([_post("Practico hipica desde que era pequena")])
        assert findings.practica_deportiva == "hipica"

    def test_detects_equitacion_as_hipica(self):
        findings = extract_demographics([_post("Hago equitacion todos los sabados")])
        assert findings.practica_deportiva == "hipica"

    def test_hipica_spectator_mention_is_not_detected(self):
        findings = extract_demographics([_post("Vi las carreras de caballos en la tele")])
        assert findings.practica_deportiva is None

    def test_detects_ajedrez_practice(self):
        findings = extract_demographics([_post("Juego al ajedrez en un club todos los jueves")])
        assert findings.practica_deportiva == "ajedrez"

    def test_ajedrez_generic_mention_is_not_detected(self):
        findings = extract_demographics([_post("El ajedrez es un deporte muy mental")])
        assert findings.practica_deportiva is None

    def test_detects_zumba_as_baile_fitness(self):
        """Regresión: en un borrador anterior 'hago zumba' caia en
        'gimnasia_intensa' -- la encuesta 2024/25 separa 'gimnasia
        intensa' de 'otra actividad fisica con musica' en dos filas
        distintas, y zumba encaja mejor en la segunda."""
        findings = extract_demographics([_post("Hago zumba los martes y jueves en el polideportivo")])
        assert findings.practica_deportiva == "baile_fitness"

    def test_baile_fitness_and_gimnasia_intensa_are_distinct(self):
        baile = extract_demographics([_post("Hago zumba los martes y jueves")])
        gimnasia = extract_demographics([_post("Voy a spinning tres veces por semana")])
        assert baile.practica_deportiva == "baile_fitness"
        assert gimnasia.practica_deportiva == "gimnasia_intensa"

    def test_no_match_leaves_none(self):
        findings = extract_demographics([_post("Hoy fui al parque con mi perro")])
        assert findings.practica_deportiva is None

    def test_first_match_wins_and_source_is_texto(self):
        findings = extract_demographics([_post("Juego al futbol y tambien voy al gimnasio")])
        assert findings.practica_deportiva == "futbol"
        assert findings.source["practica_deportiva"] == "texto"
        assert findings.evidence["practica_deportiva"] == ["https://x/1"]
