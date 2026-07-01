"""
Modelos de datos. Todo esto vive en memoria durante la petición HTTP,
nunca se escribe a disco ni a una base de datos (diseño RGPD del TFG).
"""
from datetime import datetime
from pydantic import BaseModel


class RedditPost(BaseModel):
    id: str
    type: str  # "post" | "comment"
    subreddit: str
    title: str | None = None
    text: str
    created_utc: datetime
    score: int
    permalink: str


class RedditProfile(BaseModel):
    username: str
    account_created_utc: datetime
    karma_post: int
    karma_comment: int
    bio: str | None = None
    posts: list[RedditPost]


class WritingFingerprint(BaseModel):
    avg_sentence_length: float
    vocabulary_richness: float  # type-token ratio
    emoji_usage_rate: float
    avg_posts_per_hour: dict[int, float]  # hora (0-23) -> proporción de actividad
    top_subreddits: list[tuple[str, int]]
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


class ExposureReport(BaseModel):
    username: str
    generated_at: datetime
    n_posts_analyzed: int
    fingerprint: WritingFingerprint
    inferred_attributes: list[InferredAttribute]
    privacy_score: PrivacyScore
    recommendations: list[str]
