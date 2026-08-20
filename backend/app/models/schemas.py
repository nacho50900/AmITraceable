"""
Modelos de datos. Todo esto vive en memoria durante la petición HTTP,
nunca se escribe a disco ni a una base de datos (diseño RGPD del TFG).

`SocialPost` / `SocialProfile` son el modelo genérico que alimenta el
pipeline de análisis (fingerprint -> inferencia -> scoring -> informe),
compartido por todas las plataformas soportadas. Cada cliente de plataforma
(`reddit_client.py`, `instagram_client.py`) es responsable de normalizar la
respuesta de su API al este modelo común:

- `group`: el equivalente más parecido a "comunidad/tema" que tenga la
  plataforma — el subreddit en Reddit, el primer hashtag del caption en
  Instagram. Se llama igual en ambos casos para que el resto del pipeline
  (fingerprinting, inferencia de atributos) no necesite saber de qué
  plataforma vienen los datos.
- `score`: proxy de "repercusión" del post — karma neto en Reddit,
  likes + comentarios en Instagram. Mismo razonamiento que `group`.

Si se añade una tercera plataforma en el futuro, solo hace falta escribir
su cliente devolviendo `SocialProfile`; el resto del pipeline no cambia.
"""
from datetime import datetime
from pydantic import BaseModel


class SocialPost(BaseModel):
    id: str
    platform: str  # "reddit" | "instagram"
    type: str  # Reddit: "post"/"comment". Instagram: "image"/"video"/"carousel_album"
    group: str  # subreddit (Reddit) o primer hashtag del caption (Instagram)
    # Todas las etiquetas del post: [subreddit] en Reddit (siempre una), o
    # TODOS los hashtags del caption en Instagram (puede haber varios y
    # perder los que no sean el primero penaliza la inferencia de atributos,
    # ver attribute_inference.py). Vacío si no aplica.
    tags: list[str] = []
    title: str | None = None
    text: str
    created_utc: datetime
    score: int
    permalink: str
    # URLs directas a CADA foto de la publicación (solo Instagram; vacío en
    # Reddit). En una publicación normal es una sola imagen; en un carrusel
    # (media_type "carousel_album") puede haber varias, y se analizan
    # TODAS, no solo la primera -- ver InstagramClient._normalize(). Se usa
    # ÚNICAMENTE para el módulo opcional de geolocalización por imagen
    # (app/vision/geolocation.py): cada imagen se descarga en memoria de
    # forma transitoria para extraer un embedding y se descarta acto
    # seguido, sin persistirse en disco ni en base de datos (coherente con
    # el diseño stateless del resto del proyecto, aunque supone una
    # excepción consciente al principio original de "no se descargan
    # imágenes" -- ver nota actualizada en instagram_client.py). Los vídeos
    # de un carrusel se excluyen (no son analizables por el modelo de
    # geolocalización, que solo procesa imágenes).
    media_urls: list[str] = []


class SocialProfile(BaseModel):
    platform: str
    username: str  # handle/alias, p.ej. "@ana_perez" -- no es necesariamente el nombre real
    account_created_utc: datetime | None = None  # Instagram no expone este dato
    bio: str | None = None
    # Nombre público que muestra la plataforma (si la API lo expone), p.ej.
    # "Ana Pérez" en Instagram. None en Reddit (no expone nombre real, solo
    # el handle, que ya va en `username`). Se usa ÚNICAMENTE como señal
    # débil para inferir sexo por convención de nombre en español -- ver
    # app/nlp/ai_attribute_extraction.py, donde se marca con una
    # procedencia distinta ("ia_nombre") y menor fiabilidad que una
    # autodeclaración explícita en texto.
    full_name: str | None = None
    # URL pública de la foto de perfil (si la plataforma la expone). Solo
    # se guarda la URL, nunca la imagen en sí -- coherente con "no hay
    # persistencia" del resto del proyecto: el navegador la carga
    # directamente desde el CDN de la plataforma, este backend no la
    # descarga ni la reenvía.
    avatar_url: str | None = None
    posts: list[SocialPost]


class WritingFingerprint(BaseModel):
    avg_sentence_length: float
    vocabulary_richness: float  # type-token ratio
    emoji_usage_rate: float
    avg_posts_per_hour: dict[int, float]  # hora (0-23) -> proporción de actividad
    top_groups: list[tuple[str, int]]  # subreddits o hashtags más frecuentes
    top_keywords: list[tuple[str, float]]
    detected_language: str


class InferredAttribute(BaseModel):
    category: str  # ej. "ubicacion", "rutina", "ocupacion", "edad_estimada"
    value: str
    confidence: float  # 0-1
    evidence: list[str]  # permalinks o fragmentos que lo justifican


class PrivacyScore(BaseModel):
    overall_score: float  # 0-100, mayor = más expuesto
    geolocation_risk: float
    identity_consistency_risk: float
    inferable_data_risk: float
    deanonymization_ease: float
    breakdown_explanation: dict[str, str]


class PopulationEstimate(BaseModel):
    attribute_label: str  # p.ej. "Sexo: mujer", "Vive en municipio: Leon"
    category: str  # sexo | edad | ubicacion | estudios | ocupacion | universidad | empresa
    remaining_population: int | None  # None si no estimable con las tablas actuales
    risk_level: str  # bajo | medio | alto | critico | no_estimable
    evidence: list[str]
    # "texto" (autodeclaración por regex) | "imagen" (geolocation.py) | "ia" (autodeclaración
    # detectada por IA) | "ia_nombre" (estimación de sexo por nombre público, no autodeclaración)
    source: str = "texto"
    note: str | None = None
    # Código estable (ver app/note_codes.py) para que el frontend traduzca
    # `note` sin parsear la frase en español; `note` se conserva tal cual
    # para logs y la descarga JSON del informe. None si este paso no lleva
    # nota.
    note_code: str | None = None
    # Valor "en crudo" del atributo, sin la plantilla de `attribute_label`
    # ya montada (p.ej. "hombre", "24", "León") -- ver docstring del campo
    # homónimo en scoring/k_anonymity.py.PopulationNarrowingStep.
    value_raw: str | None = None
    # Solo relevante cuando category == "ubicacion": "municipio" | "provincia"
    # | "comunidad_autonoma" -- distingue qué plantilla de traducción usar
    # en el frontend (los tres comparten category="ubicacion").
    location_level: str | None = None
    # remaining_population / TOTAL_POPULATION_ES -- ya calculado en el backend para
    # que el frontend no tenga que conocer esa constante (pictograma de población).
    proportion: float | None = None
    # % que este rasgo concreto ha reducido la población respecto al
    # escalón ANTERIOR de la cadena de estrechamiento (no respecto al total
    # de España) -- ver scoring/k_anonymity.py. None en pasos no estimables
    # y en los standalone (universidad/empresa).
    reduction_percent: float | None = None
    # Confianza (0-1) declarada por la IA para una estimación INDIRECTA,
    # cuando aplique (de momento, solo el tramo de edad estimado -- ver
    # app/scoring/k_anonymity.py::_step_edad). None para el resto de pasos,
    # incluidas las autodeclaraciones exactas (esas no llevan confianza:
    # o se detectan literalmente, o no se detectan).
    confidence: float | None = None


class ImageLocationPoint(BaseModel):
    permalink: str  # enlace a la foto que generó esta estimación (o la URL directa de la foto de perfil, ver is_profile_picture)
    province: str
    confidence: float  # 0-1, proporción de vecinos del índice que coincidieron
    lat: float | None
    lon: float | None
    # False cuando los vecinos más parecidos de la foto están repartidos por
    # una zona demasiado amplia (ver ImageLocationEstimate.representative en
    # app/vision/geolocation.py) -- la estimación es real, pero no fiable
    # como para representarla en el mapa. El frontend la muestra aparte, en
    # un apartado de "imágenes no representativas" con solo el enlace a la
    # publicación, sin pintarla como punto en el mapa.
    representative: bool = True
    # True si esta entrada es la foto de perfil, no una publicación (ver
    # `estimate_locations_for_posts` en app/vision/geolocation.py). El
    # frontend la etiqueta como "Foto de perfil" en vez de "Ver
    # publicación", y `permalink` en ese caso es la URL directa de la
    # imagen (no hay página de publicación a la que enlazar). Se excluye
    # del cálculo de consenso de residencia (ver
    # report/generator.py::_apply_image_geolocation) igual que las fotos
    # de viaje: no es necesariamente representativa de dónde vive la
    # persona.
    is_profile_picture: bool = False
    # Fecha de la publicación (post.created_utc), para poder distinguir a
    # simple vista fotos que si no fuera por esto se verían todas iguales en
    # el listado (sobre todo las "no representativas", que solo muestran el
    # enlace). None si por algún motivo no se pudo emparejar el permalink
    # con su publicación original (no debería ocurrir en la práctica).
    created_utc: datetime | None = None
    # Respuesta cruda de Moondream2 sobre el CONTENIDO de la foto (ver
    # app/vision/scene_analysis.py), no una redacción -- mismas cuatro
    # líneas (DESCRIPCION/PERSONAS/AFICION/PAREJA) que ya alimentan
    # `inferred_attributes` y el estado civil, pero aquí visibles tal
    # cual, foto por foto, para que el frontend las pueda mostrar al
    # desplegar cada foto. None si el modelo no estaba disponible en este
    # servidor, o la inferencia falló para esta foto en concreto (no
    # bloquea el resto del análisis).
    #
    # Al formar parte de este modelo, se incluye automáticamente en el JSON
    # completo del informe que ai_analysis.py le manda a Mistral para las
    # conclusiones finales -- sin necesitar ningún cambio ahí.
    visual_description: str | None = None
    # Descripción GENERAL de la escena (campo DESCRIPCION del prompt de
    # scene_analysis.py, ya parseado -- p. ej. "4 personas comiendo pizza
    # alegremente en una terraza"). A diferencia de `visual_description`
    # (las cuatro líneas crudas, para la vista de detalle "qué vio la
    # IA"), este campo es una frase legible pensada para mostrarse de
    # forma prominente como pie de foto. Nunca menciona raza, etnia, tono
    # de piel, edad ni aspecto físico de ninguna persona -- restricción
    # explícita del prompt (ver scene_analysis.py), no un filtro aparte.
    # None en los mismos casos que `visual_description`.
    visual_description_general: str | None = None


class ExposureReport(BaseModel):
    platform: str
    username: str
    generated_at: datetime
    n_posts_analyzed: int
    fingerprint: WritingFingerprint
    inferred_attributes: list[InferredAttribute]
    privacy_score: PrivacyScore
    recommendations: list[str]
    # Estrechamiento progresivo de población compatible con cada atributo
    # autodeclarado detectado (k-anonimato aproximado, ver scoring/k_anonymity.py).
    # Lista vacía si no se detectó ninguna declaración explícita en el texto.
    population_narrowing: list[PopulationEstimate] = []
    # Nº de personas en España que compartirían, EN CONJUNTO, todos los
    # rasgos encadenables detectados (sexo + edad + ubicación + estudios +
    # ocupación) -- no el porcentaje de un rasgo aislado, sino la
    # intersección de todos los que se hayan podido estimar. Ver
    # scoring/k_anonymity.py::final_remaining_population(). None si no se
    # pudo estimar ningún rasgo encadenable (no hay número que mostrar).
    remaining_population_all_traits: int | None = None
    # remaining_population_all_traits / TOTAL_POPULATION_ES, ya calculado
    # aquí (mismo criterio que PopulationEstimate.proportion) para que el
    # frontend no tenga que conocer ni duplicar esa constante -- usado para
    # el ÚNICO pictograma grande que resume "Qué se puede inferir sobre ti"
    # (ver PopulationNarrowingTable.tsx / Dashboard.tsx). None si
    # remaining_population_all_traits también lo es.
    remaining_population_all_traits_proportion: float | None = None
    # Una estimación de ubicación por CADA foto analizada por
    # app/vision/geolocation.py (no solo la de mayor confianza, que es la
    # única que se usa para population_narrowing). Se usa para pintar el
    # mapa de puntos en el frontend. Vacía si no hay índice FAISS construido
    # o la plataforma no es Instagram.
    image_location_points: list[ImageLocationPoint] = []
    # False si el índice FAISS de geolocalización no está construido en
    # este servidor (o faltan sus dependencias opcionales), o la
    # plataforma no es Instagram. Sirve para que el frontend distinga ese
    # caso ("la función no está disponible aquí") de "se analizaron fotos
    # pero ninguna dio una estimación fiable" -- son mensajes distintos.
    geolocation_available: bool = False
    # Foto de perfil pública de la cuenta analizada (si la plataforma la
    # expone), para identificar visualmente de quién es el informe en el
    # título del dashboard. Solo la URL -- ver nota en SocialProfile.
    avatar_url: str | None = None
