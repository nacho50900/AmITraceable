import type { ExposureReport } from '../types';

/** Construye un ExposureReport completo y válido para tests, con valores de
 * ejemplo razonables. Acepta overrides parciales para los tests que
 * necesiten variar un campo concreto (plataforma, usuario, fecha...). */
export function makeExposureReport(overrides: Partial<ExposureReport> = {}): ExposureReport {
  return {
    platform: 'instagram',
    username: 'usuario_prueba',
    generated_at: '2026-07-08T12:00:00+00:00',
    n_posts_analyzed: 5,
    fingerprint: {
      avg_sentence_length: 12.3,
      vocabulary_richness: 0.55,
      emoji_usage_rate: 0.02,
      avg_posts_per_hour: { '0': 0, '12': 2, '20': 3 },
      top_groups: [['madrid', 3]],
      top_keywords: [['viajes', 4]],
      detected_language: 'es',
    },
    inferred_attributes: [
      {
        category: 'ubicacion',
        value: 'Posible vínculo geográfico con: madrid',
        confidence: 0.7,
        evidence: ['https://instagram.com/p/1'],
      },
    ],
    privacy_score: {
      overall_score: 42.5,
      geolocation_risk: 30,
      identity_consistency_risk: 0,
      inferable_data_risk: 50,
      deanonymization_ease: 20,
      breakdown_explanation: {
        geolocation: 'Basado en menciones y hashtags geolocalizables detectados.',
        identity_consistency: 'No evaluado en esta versión.',
        inferable_data: 'Basado en número y confianza de atributos personales inferidos.',
        deanonymization_ease: 'Basado en consistencia temporal de actividad.',
      },
    },
    recommendations: ['Evita etiquetar la ubicación exacta.', 'Revisa qué hashtags revelan tu rutina.'],
    population_narrowing: [
      {
        attribute_label: 'Sexo: mujer',
        category: 'sexo',
        remaining_population: 24957175,
        risk_level: 'bajo',
        evidence: ['https://instagram.com/p/1'],
        source: 'texto',
        note: null,
      },
    ],
    image_location_points: [
      { permalink: 'https://instagram.com/p/1', province: 'Madrid', confidence: 0.6, lat: 40.41, lon: -3.7 },
    ],
    ...overrides,
  };
}
