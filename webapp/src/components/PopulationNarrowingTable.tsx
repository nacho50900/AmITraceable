import React from 'react';
import type { PopulationEstimate } from '../types';

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
};

const SOURCE_ICONS: Record<string, string> = {
  texto: '✍️',
  imagen: '📷',
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
      <table className="population-narrowing-table">
        <thead>
          <tr>
            <th>Información detectada</th>
            <th>Población restante (aprox.)</th>
            <th>Riesgo</th>
            <th>Fuente</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((step, i) => (
            <tr key={i}>
              <td>
                {step.attribute_label}
                {step.note && <div className="note-inline">{step.note}</div>}
              </td>
              <td>{formatPopulation(step.remaining_population)}</td>
              <td>
                <span
                  className="risk-pill"
                  style={{ background: RISK_COLORS[step.risk_level], color: '#fff' }}
                >
                  {RISK_LABELS[step.risk_level]}
                </span>
              </td>
              <td>
                <span className="source-badge" title={step.source === 'imagen' ? 'Estimado a partir de una imagen (menor fiabilidad que una autodeclaración de texto)' : 'Detectado en texto que escribiste tú mismo/a'}>
                  {SOURCE_ICONS[step.source]} {SOURCE_LABELS[step.source]}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
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
      </p>
    </>
  );
};

export default PopulationNarrowingTable;
