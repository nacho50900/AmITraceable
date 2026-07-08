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
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        Estimación aproximada a partir de distribuciones agregadas del INE, asumiendo
        independencia entre atributos. No es un recuento exacto de personas.
      </p>
    </>
  );
};

export default PopulationNarrowingTable;
