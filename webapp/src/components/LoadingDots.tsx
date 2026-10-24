import React from 'react';

/**
 * Tres puntos suspensivos animados ("..."): van apareciendo uno a uno y
 * "saltando" mientras algo carga. Son decorativos (`aria-hidden`): el texto
 * accesible es la frase que los acompaña, sin los puntos. La animación está
 * en CSS (`.loading-dots`, ver index.css) y se desactiva con
 * `prefers-reduced-motion`, dejando los tres puntos fijos.
 */
const LoadingDots: React.FC = () => (
  <span className="loading-dots" aria-hidden="true">
    <span>.</span>
    <span>.</span>
    <span>.</span>
  </span>
);

export default LoadingDots;
