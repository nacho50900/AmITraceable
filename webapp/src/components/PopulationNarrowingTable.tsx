import React from 'react';
import type { PopulationEstimate } from '../types';
import PopulationPictogram from './PopulationPictogram';

interface PopulationNarrowingTableProps {
  steps: PopulationEstimate[];
}

const RISK_COLORS: Record<string, string> = {
  bajo: '#3aa657',
  medio: '#d6a51c',
  alto: '#e0792f',
  critico: '#d3403a',
  no_estimable: '#8a8a8a',
};

const RISK_LABELS: Record<string, string> = {
  bajo: 'Bajo',
  medio: 'Medio',
  alto: 'Alto',
  critico: 'Crítico',
  no_estimable: 'No estimable',
};

const SOURCE_LABELS: Record<string, string> = {
  texto: 'Texto',
  imagen: 'Imagen',
  ia: 'IA (texto)',
  ia_nombre: 'IA (nombre)',
};

const SOURCE_ICONS: Record<string, string> = {
  texto: '✍️',
  imagen: '📷',
  ia: '🤖',
  ia_nombre: '🤖',
};

const SOURCE_TITLES: Record<string, string> = {
  texto: 'Detectado en texto que escribiste tú mismo/a',
  imagen: 'Estimado a partir de una imagen (menor fiabilidad que una autodeclaración de texto)',
  ia: 'Detectado por un modelo de IA en texto/biografía que escribiste tú mismo/a',
  ia_nombre:
    'Estimado por convención cultural del nombre público de tu cuenta, no por algo que hayas escrito -- fiabilidad menor',
};

function formatPopulation(value: number | null): string {
  if (value === null) return '—';
  return value.toLocaleString('es-ES');
}

const PopulationNarrowingTable: React.FC<PopulationNarrowingTableProps> = ({ steps }) => {
  if (steps.length === 0) {
    return (
      <p className="note">
        No se han detectado declaraciones explícitas sobre ti (edad, sexo, ubicación,
        estudios...) en tu texto público, así que no hay una estimación de población que
        mostrar aquí.
      </p>
    );
  }

  return (
    <>
      <div className="population-narrowing-list">
        {steps.map((step) => (
          <div className="population-step" key={`${step.category}-${step.attribute_label}`}>
            <div className="population-step-header">
              <strong>{step.attribute_label}</strong>
              <div className="population-step-badges">
                <span
                  className="risk-pill"
                  style={{ background: RISK_COLORS[step.risk_level], color: '#fff' }}
                >
                  {RISK_LABELS[step.risk_level]}
                </span>
                <span className="source-badge" title={SOURCE_TITLES[step.source]}>
                  {SOURCE_ICONS[step.source]} {SOURCE_LABELS[step.source]}
                </span>
              </div>
            </div>

            {step.note && <p className="note-inline">{step.note}</p>}

            <div className="population-step-visual">
              {step.proportion !== null && step.remaining_population !== null ? (
                <PopulationPictogram
                  proportion={step.proportion}
                  remainingPopulation={step.remaining_population}
                  size="large"
                />
              ) : (
                <span className="population-fallback">{formatPopulation(step.remaining_population)}</span>
              )}
            </div>
          </div>
        ))}
      </div>
      <p className="note">
        Estimación aproximada a partir de distribuciones agregadas del INE, asumiendo
        independencia entre atributos. No es un recuento exacto de personas.
        {steps.some((s) => s.source === 'imagen') && (
          <>
            {' '}
            Las filas marcadas con 📷 vienen de un análisis visual de tus fotos, no de algo
            que hayas escrito — son menos fiables que una autodeclaración de texto.
          </>
        )}
        {steps.some((s) => s.source === 'ia_nombre') && (
          <>
            {' '}
            Las filas marcadas con 🤖 (nombre) son una estimación por el nombre público de tu
            cuenta, no algo que hayas declarado — la fiabilidad es menor.
          </>
        )}
      </p>
    </>
  );
};

export default PopulationNarrowingTable;
