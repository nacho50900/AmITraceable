import React from 'react';
import { useTranslation } from 'react-i18next';
import type { PopulationEstimate } from '../types';
import PopulationPictogram from './PopulationPictogram';

interface PopulationNarrowingTableProps {
  steps: PopulationEstimate[];
  // El ÚNICO pictograma grande de esta sección resume TODOS los rasgos
  // combinados (no cada fila por separado, como antes) -- ver
  // scoring/k_anonymity.py::final_remaining_population(). null si no se
  // pudo estimar ningún rasgo encadenable.
  remainingPopulationAllTraits: number | null;
  remainingPopulationAllTraitsProportion: number | null;
}

const RISK_COLORS: Record<string, string> = {
  bajo: '#3aa657',
  medio: '#d6a51c',
  alto: '#e0792f',
  critico: '#d3403a',
  no_estimable: '#8a8a8a',
};

function formatPopulation(value: number | null, locale: string): string {
  if (value === null) return '—';
  return value.toLocaleString(locale);
}

// `category` + `value_raw` (+ `location_level` solo para ubicación) le
// llegan del backend separados desde app/scoring/k_anonymity.py
// (PopulationNarrowingStep) precisamente para poder traducir la plantilla
// aquí sin tocar el valor cuando ese valor es un nombre propio (topónimo,
// universidad, empresa). Si el backend es una versión anterior sin estos
// campos (`value_raw` ausente), se usa `attribute_label` tal cual, ya en
// español, como fallback -- nunca se intenta parsear esa frase.
function buildAttributeLabel(
  step: PopulationEstimate,
  t: (key: string, opts?: Record<string, unknown>) => string,
  i18nExists: (key: string) => boolean,
): string {
  if (!step.value_raw) return step.attribute_label;

  const templateKey =
    step.category === 'ubicacion' && step.location_level
      ? `ubicacion_${step.location_level}`
      : step.category;
  const templateI18nKey = `dashboard.attribute.${templateKey}`;
  if (!i18nExists(templateI18nKey)) return step.attribute_label;

  const valueI18nKey = `dashboard.attributeValue.${step.category}.${step.value_raw}`;
  // Solo se traduce el VALOR cuando es de un conjunto cerrado con entrada
  // en el diccionario (sexo, estado civil, estudios...) -- nombres propios
  // (universidad, empresa, topónimos) y números (edad) se interpolan tal
  // cual, sin buscar traducción.
  const value = i18nExists(valueI18nKey) ? t(valueI18nKey) : step.value_raw;

  return t(templateI18nKey, { value });
}

function buildNote(
  step: PopulationEstimate,
  t: (key: string) => string,
  i18nExists: (key: string) => boolean,
): string | null {
  if (!step.note_code) return step.note;
  const noteI18nKey = `dashboard.noteCodes.${step.note_code}`;
  return i18nExists(noteI18nKey) ? t(noteI18nKey) : step.note;
}

const PopulationNarrowingTable: React.FC<PopulationNarrowingTableProps> = ({
  steps,
  remainingPopulationAllTraits,
  remainingPopulationAllTraitsProportion,
}) => {
  const { t, i18n } = useTranslation();
  // Los números se formatean en el idioma de la UI (separadores de miles,
  // etc.); el dato en sí sigue viniendo del INE (población española).
  const numberLocale = i18n.language?.split('-')[0] === 'en' ? 'en-US' : 'es-ES';

  if (steps.length === 0) {
    return <p className="note">{t('components.populationNarrowing.noData')}</p>;
  }

  return (
    <>
      <div className="population-narrowing-list">
        {steps.map((step) => {
          const label = buildAttributeLabel(step, t, i18n.exists.bind(i18n));
          const note = buildNote(step, t, i18n.exists.bind(i18n));
          return (
            <div className="population-step" key={`${step.category}-${step.attribute_label}`}>
              <div className="population-step-header">
                <strong>{label}</strong>
                <div className="population-step-badges">
                  {step.reduction_percent !== null && (
                    <span
                      className="reduction-badge"
                      title={t('components.populationNarrowing.reductionBadgeTitle')}
                    >
                      -{step.reduction_percent}%
                    </span>
                  )}
                  {step.confidence !== null && step.confidence !== undefined && (
                    <span
                      className="confidence-badge"
                      title={t('components.populationNarrowing.confidenceBadgeTitle')}
                    >
                      ~{Math.round(step.confidence * 100)}%
                    </span>
                  )}
                  <span
                    className="risk-pill"
                    style={{ background: RISK_COLORS[step.risk_level], color: '#fff' }}
                  >
                    {t(`components.populationNarrowing.risk.${step.risk_level}`)}
                  </span>
                  <span
                    className={`source-badge${step.source === 'manual' ? ' source-badge--manual' : ''}`}
                    title={
                      i18n.exists(`components.populationNarrowing.sourceTitle.${step.source}`)
                        ? t(`components.populationNarrowing.sourceTitle.${step.source}`)
                        : t('components.populationNarrowing.sourceTitle.manual')
                    }
                  >
                    {SOURCE_ICONS[step.source]}{' '}
                    {i18n.exists(`components.populationNarrowing.source.${step.source}`)
                      ? t(`components.populationNarrowing.source.${step.source}`)
                      : t('components.populationNarrowing.source.manual')}
                  </span>
                </div>
              </div>

              {note && <p className="note-inline">{note}</p>}

              <span className="population-fallback">{formatPopulation(step.remaining_population, numberLocale)}</span>
            </div>
          );
        })}
      </div>

      {remainingPopulationAllTraits !== null && (
        <div className="population-combined-summary">
          <p className="trait-summary">
            {t('components.populationNarrowing.combinedSummaryPrefix')}{' '}
            <span className="trait-summary-number">
              {remainingPopulationAllTraits.toLocaleString(numberLocale)}
            </span>{' '}
            {t('components.populationNarrowing.combinedSummarySuffix')}
          </p>
          <PopulationPictogram
            proportion={remainingPopulationAllTraitsProportion}
            remainingPopulation={remainingPopulationAllTraits}
            size="large"
          />
        </div>
      )}
      <p className="note">
        {t('components.populationNarrowing.footnote')}
        {steps.some((s) => s.source === 'imagen') && t('components.populationNarrowing.footnoteImage')}
        {steps.some((s) => s.source === 'ia_nombre') && t('components.populationNarrowing.footnoteNameGuess')}
      </p>
    </>
  );
};

const SOURCE_ICONS: Record<string, string> = {
  texto: '✍️',
  imagen: '📷',
  ia: '🤖',
  ia_nombre: '🤖',
  ia_simbolica: '🤖',
  ia_estimada: '🤖',
  manual: '👤',
};

export default PopulationNarrowingTable;
