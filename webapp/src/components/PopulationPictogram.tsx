import React from 'react';
import { useTranslation } from 'react-i18next';

interface PopulationPictogramProps {
  /** Fracción 0-1 (remaining_population / población total de España), ya
   * calculada en el backend -- ver app/scoring/k_anonymity.py. */
  proportion: number | null;
  remainingPopulation: number | null;
  /** 'large' para un bloque visual destacado (una fila por atributo, ver
   * PopulationNarrowingTable): monigotes grandes de verdad cuando hay
   * pocos (p.ej. 250px+ para un 50%, que son solo 2). 'small' para usos
   * más compactos con muchos monigotes. */
  size?: 'small' | 'large';
}

// Si tocaría dibujar más de este número de monigotes para representar la
// proporción exacta (proporciones muy pequeñas), la rejilla deja de ser
// legible -- se cambia a un único monigote turquesa + "de X" con el
// número absoluto de personas.
const MAX_FIGURES = 100;

/** Tamaño (px) de cada monigote en modo grande: cuantos MENOS monigotes
 * haga falta dibujar, MÁS grande se ve cada uno -- para un 50% (2
 * monigotes) deben verse genuinamente grandes, no un icono minúsculo. */
function largeFigureSize(totalFigures: number): number {
  if (totalFigures <= 2) return 260;
  if (totalFigures <= 4) return 180;
  if (totalFigures <= 10) return 110;
  if (totalFigures <= 25) return 64;
  if (totalFigures <= 50) return 40;
  return 26;
}

/**
 * Representación visual "isotype" (un monigote = una porción de la
 * población): para un 50%, 1 monigote turquesa + 1 negro; para un 10%, 1
 * turquesa + 9 negros; etc. Para proporciones por debajo de aprox. 1%
 * (donde tocaría dibujar más de MAX_FIGURES monigotes), se simplifica a un
 * único monigote turquesa junto al número absoluto de personas, que es más
 * legible que una rejilla enorme.
 */
const PopulationPictogram: React.FC<PopulationPictogramProps> = ({
  proportion,
  remainingPopulation,
  size = 'small',
}) => {
  const { t, i18n } = useTranslation();
  const numberLocale = i18n.language?.split('-')[0] === 'en' ? 'en-US' : 'es-ES';

  if (proportion === null || proportion <= 0 || remainingPopulation === null) {
    return null;
  }

  const percentLabel = proportion * 100 >= 1 ? `${Math.round(proportion * 100)}%` : `${(proportion * 100).toFixed(2)}%`;
  const totalFigures = Math.round(1 / proportion);
  const populationLabel = remainingPopulation.toLocaleString(numberLocale);
  const isLarge = size === 'large';

  if (totalFigures > MAX_FIGURES) {
    const compactSize = isLarge ? 220 : 20;
    // El icono compacto representa la PROPORCIÓN ("1 de cada X"), no el
    // total absoluto de personas -- ese total ya se muestra por separado
    // encima, en la frase "En España hay X personas..." (ver
    // PopulationNarrowingTable.tsx). Mostrar aquí también el total
    // absoluto (p. ej. "de 17.838") es tanto redundante como
    // semánticamente incorrecto: el monigote no representa "eres 1 de
    // 17.838 personas", representa "1 de cada X españoles al azar tiene
    // tus rasgos" -- X es `totalFigures` (1/proportion), no
    // `remainingPopulation`.
    const ratioLabel = totalFigures.toLocaleString(numberLocale);
    return (
      <div
        className={`pictogram pictogram-compact ${isLarge ? 'pictogram-lg' : ''}`}
        role="img"
        aria-label={t('components.populationPictogram.compactAriaLabel', {
          percent: percentLabel,
          ratio: ratioLabel,
        })}
      >
        <img
          src="/monigote_selected.png"
          alt=""
          className="pictogram-figure"
          style={{ width: compactSize, height: compactSize }}
        />
        <span className="pictogram-compact-label">
          {t('components.populationPictogram.compactLabel', { ratio: ratioLabel })}
        </span>
      </div>
    );
  }

  const figureSize = isLarge ? largeFigureSize(totalFigures) : 14;

  return (
    <div
      className={`pictogram ${isLarge ? 'pictogram-lg' : ''}`}
      role="img"
      aria-label={t('components.populationPictogram.gridAriaLabel', {
        percent: percentLabel,
        population: populationLabel,
      })}
    >
      <span className="pictogram-percent">{percentLabel}</span>
      <div className="pictogram-grid">
        {Array.from({ length: totalFigures }, (_, i) => (
          <img
            key={i}
            src={i === 0 ? '/monigote_selected.png' : '/monigote.png'}
            alt=""
            className="pictogram-figure"
            style={{ width: figureSize, height: figureSize }}
          />
        ))}
      </div>
    </div>
  );
};

export default PopulationPictogram;
