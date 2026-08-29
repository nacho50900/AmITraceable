"""
Códigos de nota para `PopulationNarrowingStep.note_code` (ver
scoring/k_anonymity.py), mismo patrón y mismo motivo que `stages.py`: el
campo `note` ya existente sigue llevando el texto explicativo completo en
español (se conserva para logs, la descarga JSON del informe y cualquier
integración que ya dependa de él), pero para la UI en inglés hace falta
algo que el frontend pueda traducir sin parsear frases -- de ahí este
código adicional, estable e independiente de idioma.

Se comprobó que TODAS las `note` de k_anonymity.py son texto fijo,
condicionado como mucho por `source` o por si se usó una tabla cruzada
exacta (nunca interpolan un valor libre del usuario) -- por eso encajan en
este mismo patrón cerrado de códigos, a diferencia de `attribute_label`
(que sí interpola nombres propios como universidad/empresa/topónimos, y
necesita separar plantilla y valor en vez de un código único).
"""

SEXO_ESTIMADO_POR_NOMBRE = "sexo_estimado_por_nombre"
NO_INE_DATA_FOR_VALUE = "no_ine_data_for_value"
ESTADO_CIVIL_IA_SIMBOLICA = "estado_civil_ia_simbolica"
ESTADO_CIVIL_IA_SIMBOLICA_EXACT_COMBO = "estado_civil_ia_simbolica_exact_combo"
EDAD_REPARTIDA_UNIFORMEMENTE = "edad_repartida_uniformemente"
EDAD_ESTIMADA_POR_TRAMO = "edad_estimada_por_tramo"
LOCATION_NO_POPULATION_DATA = "location_no_population_data"
LOCATION_NOTE_BASE = "location_note_base"
LOCATION_NOTE_IMAGEN = "location_note_imagen"
SITUACION_LABORAL_NOTE = "situacion_laboral_note"
TIPO_HOGAR_NOTE = "tipo_hogar_note"
LENGUA_NO_ESTIMABLE = "lengua_no_estimable"
LENGUA_WITHIN_CCAA = "lengua_within_ccaa"
UNIVERSIDAD_NO_ESTIMABLE = "universidad_no_estimable"
EMPRESA_NO_ESTIMABLE = "empresa_no_estimable"
ORIENTACION_SEXUAL_CATEGORIA_ESPECIAL = "orientacion_sexual_categoria_especial"
RELIGION_CATEGORIA_ESPECIAL = "religion_categoria_especial"
SIGNO_ZODIACAL_NOTE = "signo_zodiacal_note"
PRACTICA_DEPORTIVA_NO_PARTICION = "practica_deportiva_no_particion"
PRACTICA_DEPORTIVA_AJUSTADA_POR_SEXO = "practica_deportiva_ajustada_por_sexo"