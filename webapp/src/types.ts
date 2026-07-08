// Estos tipos reflejan 1:1 los modelos Pydantic de `users/app/models/schemas.py`.
// Si cambia el backend, hay que actualizar este fichero a mano (no hay
// generación automática de tipos en este MVP; queda como mejora futura,
// por ejemplo con openapi-typescript sobre el /openapi.json de FastAPI).

export type Platform = 'reddit' | 'instagram';

export interface WritingFingerprint {
  avg_sentence_length: number;
  vocabulary_richness: number;
  emoji_usage_rate: number;
  avg_posts_per_hour: Record<string, number>;
  top_groups: [string, number][];
  top_keywords: [string, number][];
  detected_language: string;
}

export interface InferredAttribute {
  category: string;
  value: string;
  confidence: number;
  evidence: string[];
}

export interface PrivacyScore {
  overall_score: number;
  geolocation_risk: number;
  identity_consistency_risk: number;
  inferable_data_risk: number;
  deanonymization_ease: number;
  breakdown_explanation: Record<string, string>;
}

export interface PopulationEstimate {
  attribute_label: string;
  category: string;
  remaining_population: number | null;
  risk_level: 'bajo' | 'medio' | 'alto' | 'critico' | 'no_estimable';
  evidence: string[];
  source: 'texto' | 'imagen';
  note: string | null;
}

export interface ImageLocationPoint {
  permalink: string;
  province: string;
  confidence: number;
  lat: number | null;
  lon: number | null;
}

export interface ExposureReport {
  platform: Platform;
  username: string;
  generated_at: string;
  n_posts_analyzed: number;
  fingerprint: WritingFingerprint;
  inferred_attributes: InferredAttribute[];
  privacy_score: PrivacyScore;
  recommendations: string[];
  population_narrowing: PopulationEstimate[];
  image_location_points: ImageLocationPoint[];
}

export interface AuthStatus {
  authenticated: boolean;
}
