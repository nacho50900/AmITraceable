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


class SocialProfile(BaseModel):
    platform: str
    username: str
    account_created_utc: datetime | None = None  # Instagram no expone este dato
    bio: str | None = None
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
    note: str | None = None


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
