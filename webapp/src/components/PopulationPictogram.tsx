import React from 'react';

interface PopulationPictogramProps {
  /** Fracción 0-1 (remaining_population / población total de España), ya
   * calculada en el backend -- ver app/scoring/k_anonymity.py. */
  proportion: number | null;
  remainingPopulation: number | null;
}

// Si tocaría dibujar más de este número de monigotes para representar la
// proporción exacta (proporciones muy pequeñas), la rejilla deja de ser
// legible -- se cambia a un único monigote turquesa + "de X" con el
// número absoluto de personas.
const MAX_FIGURES = 100;

/**
 * Representación visual "isotype" (un monigote = una porción de la
 * población): para un 50%, 1 monigote turquesa + 1 negro; para un 10%, 1
 * turquesa + 9 negros; etc. Para proporciones por debajo de aprox. 1%
 * (donde tocaría dibujar más de MAX_FIGURES monigotes), se simplifica a un
 * único monigote turquesa junto al número absoluto de personas, que es más
 * legible que una rejilla enorme.
 */
const PopulationPictogram: React.FC<PopulationPictogramProps> = ({ proportion, remainingPopulation }) => {
  if (proportion === null || proportion <= 0 || remainingPopulation === null) {
    return null;
  }

  const percentLabel = proportion * 100 >= 1 ? `${Math.round(proportion * 100)}%` : `${(proportion * 100).toFixed(2)}%`;
  const totalFigures = Math.round(1 / proportion);
  const populationLabel = remainingPopulation.toLocaleString('es-ES');

  if (totalFigures > MAX_FIGURES) {
    return (
      <div
        className="pictogram pictogram-compact"
        role="img"
        aria-label={`Aproximadamente ${percentLabel} de la población: 1 de cada ${populationLabel} personas`}
      >
        <img src="/monigote-selected.png" alt="" className="pictogram-figure" />
        <span className="pictogram-compact-label">de {populationLabel}</span>
      </div>
    );
  }

  return (
    <div
      className="pictogram"
      role="img"
      aria-label={`Aproximadamente ${percentLabel} de la población comparte esta característica (${populationLabel} personas)`}
    >
      <span className="pictogram-percent">{percentLabel}</span>
      <div className="pictogram-grid">
        {Array.from({ length: totalFigures }, (_, i) => (
          <img
            key={i}
            src={i === 0 ? '/monigote-selected.png' : '/monigote.png'}
            alt=""
            className="pictogram-figure"
          />
        ))}
      </div>
    </div>
  );
};

export default PopulationPictogram;
