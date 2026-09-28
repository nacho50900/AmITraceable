import React from 'react';
import './Skeleton.css';

// Placeholders animados que se muestran mientras carga el contenido real.
// Son puramente decorativos: van con aria-hidden y el estado de carga se
// comunica con el texto que ya existe en cada pantalla (p.ej.
// dashboard.analyzing) y con aria-busy en el contenedor de la página.

interface SkeletonProps {
  readonly width?: string | number;
  readonly height?: string | number;
  readonly radius?: string | number;
  readonly className?: string;
}

/** Bloque base. Sin `width`/`height`/`radius` explícitos, manda la clase CSS
 * (`.skeleton` o la que se pase en `className`). */
export const Skeleton: React.FC<SkeletonProps> = ({ width, height, radius, className }) => (
  <span
    className={className ? `skeleton ${className}` : 'skeleton'}
    style={{ width, height, borderRadius: radius }}
    aria-hidden="true"
  />
);

// Ids estables para las listas de placeholders (son contenido estático, no
// datos: no hay un identificador "real" del que colgar la `key`).
function placeholderIds(count: number, prefix: string): string[] {
  return Array.from({ length: count }, (_, i) => `${prefix}-${i}`);
}

interface SkeletonTextProps {
  readonly lines?: number;
  readonly lastLineWidth?: string;
}

/** Párrafo de `lines` líneas; la última, más corta, como en un texto real. */
export const SkeletonText: React.FC<SkeletonTextProps> = ({ lines = 3, lastLineWidth = '60%' }) => {
  const ids = placeholderIds(lines, 'line');
  return (
    <div className="skeleton-text" aria-hidden="true">
      {ids.map((id, i) => (
        <Skeleton key={id} height={14} width={lines > 1 && i === lines - 1 ? lastLineWidth : '100%'} />
      ))}
    </div>
  );
};

const DOT_IDS = placeholderIds(3, 'dot');
const SCORE_BAR_IDS = placeholderIds(3, 'score-bar');
const TABLE_ROW_IDS = placeholderIds(4, 'table-row');

/** Landing: sustituye al selector de plataformas mientras se comprueba si
 * ya hay sesión (hint + mazo con flechas + puntos + botón). */
export const PlatformPickerSkeleton: React.FC = () => (
  <div className="skeleton-picker" aria-hidden="true" data-testid="platform-picker-skeleton">
    <Skeleton className="skeleton-picker-hint" />
    <div className="skeleton-deck-row">
      <Skeleton className="skeleton-arrow" />
      <Skeleton className="skeleton-deck-card" />
      <Skeleton className="skeleton-arrow" />
    </div>
    <div className="skeleton-dots">
      {DOT_IDS.map((id) => (
        <Skeleton key={id} className="skeleton-dot" />
      ))}
    </div>
    <Skeleton className="skeleton-cta" />
  </div>
);

function SkeletonCard({ children }: { readonly children: React.ReactNode }) {
  return (
    <section className="card">
      <Skeleton className="skeleton-card-title" />
      {children}
    </section>
  );
}

/** Dashboard: esqueleto del informe (puntuación, atributos inferibles,
 * mapa, gráfico horario y resumen de IA), mostrado bajo la lista de
 * progreso mientras el análisis está en curso. */
export const DashboardSkeleton: React.FC = () => (
  <div className="skeleton-dashboard" aria-hidden="true" data-testid="dashboard-skeleton">
    <SkeletonCard>
      {SCORE_BAR_IDS.map((id) => (
        <div key={id} className="skeleton-bar-row">
          <Skeleton width="35%" height={14} />
          <Skeleton className="skeleton-bar" />
        </div>
      ))}
    </SkeletonCard>

    <SkeletonCard>
      {TABLE_ROW_IDS.map((id) => (
        <div key={id} className="skeleton-table-row">
          <Skeleton width="30%" height={14} />
          <Skeleton width="45%" height={14} />
          <Skeleton width="15%" height={14} />
        </div>
      ))}
    </SkeletonCard>

    <SkeletonCard>
      <Skeleton className="skeleton-map" />
    </SkeletonCard>

    <SkeletonCard>
      <Skeleton className="skeleton-chart" />
    </SkeletonCard>

    <SkeletonCard>
      <SkeletonText lines={3} />
    </SkeletonCard>
  </div>
);
