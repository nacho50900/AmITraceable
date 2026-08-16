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
        {steps.map((step) => (
          <div className="population-step" key={`${step.category}-${step.attribute_label}`}>
            <div className="population-step-header">
              {/* attribute_label lo genera el backend (categoría inferida en
                  español); traducirlo queda fuera del alcance de esta fase. */}
              <strong>{step.attribute_label}</strong>
              <div className="population-step-badges">
                {step.reduction_percent !== null && (
                  <span
                    className="reduction-badge"
                    title={t('components.populationNarrowing.reductionBadgeTitle')}
                  >
                    -{step.reduction_percent}%
                  </span>
                )}
                <span
                  className="risk-pill"
                  style={{ background: RISK_COLORS[step.risk_level], color: '#fff' }}
                >
                  {t(`components.populationNarrowing.risk.${step.risk_level}`)}
                </span>
                <span
                  className="source-badge"
                  title={t(`components.populationNarrowing.sourceTitle.${step.source}`)}
                >
                  {SOURCE_ICONS[step.source]} {t(`components.populationNarrowing.source.${step.source}`)}
                </span>
              </div>
            </div>

            {/* step.note también viene del backend, ya en español. */}
            {step.note && <p className="note-inline">{step.note}</p>}

            <span className="population-fallback">{formatPopulation(step.remaining_population, numberLocale)}</span>
          </div>
        ))}
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
};

export default PopulationNarrowingTable;
