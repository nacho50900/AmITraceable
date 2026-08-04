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
  source: 'texto' | 'imagen' | 'ia' | 'ia_nombre' | 'ia_simbolica';
  note: string | null;
  // remaining_population / poblacion total de España, ya calculado en el
  // backend (ver app/scoring/k_anonymity.py) para no duplicar esa
  // constante aquí. null si remaining_population también lo es.
  proportion: number | null;
  // % que ESTE rasgo concreto ha reducido la población respecto al
  // escalón ANTERIOR de la cadena (no respecto al total de España) -- ver
  // app/scoring/k_anonymity.py. null en pasos no estimables y en los
  // standalone (universidad/empresa).
  reduction_percent: number | null;
}

export interface ImageLocationPoint {
  permalink: string;
  province: string;
  confidence: number;
  lat: number | null;
  lon: number | null;
  // false si los vecinos más parecidos de la foto están repartidos por una
  // zona demasiado amplia como para que la estimación sea fiable -- no se
  // pinta en el mapa, se muestra aparte en un apartado de "imágenes no
  // representativas" (ver LocationMap.tsx).
  representative: boolean;
  // Fecha de la publicación (ISO 8601), para poder distinguir las fotos del
  // listado entre sí -- sobre todo las "no representativas", que si no
  // fuera por esto se verían todas iguales (solo un enlace). null si no se
  // pudo emparejar (no debería ocurrir en la práctica).
  created_utc: string | null;
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
  // Nº de personas en España que comparten, EN CONJUNTO, todos los rasgos
  // encadenables detectados (sexo + edad + ubicación + estudios +
  // ocupación) -- no el porcentaje de un rasgo aislado. null si no se pudo
  // estimar ningún rasgo encadenable. Ver k_anonymity.py::final_remaining_population().
  remaining_population_all_traits: number | null;
  // remaining_population_all_traits / población total de España, ya
  // calculada en el backend -- para el único pictograma grande que resume
  // toda la sección (ver PopulationNarrowingTable.tsx).
  remaining_population_all_traits_proportion: number | null;
  image_location_points: ImageLocationPoint[];
  // false si el índice de geolocalización no está construido en este
  // servidor (o la plataforma no es Instagram) -- distinto de "se
  // analizaron fotos pero ninguna dio resultado fiable".
  geolocation_available: boolean;
}

// Eventos que llegan por /api/analyze/{platform}/stream (Server-Sent Events).
export type AnalysisProgressEvent =
  | { done: false; stage: string; [key: string]: unknown }
  | { done: true; report: ExposureReport }
  | { done: true; error: string };

export interface AuthStatus {
  authenticated: boolean;
}
