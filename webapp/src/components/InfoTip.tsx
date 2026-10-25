import React, { useEffect, useId, useRef, useState } from 'react';

interface InfoTipProps {
  /** Texto de la descripción que se muestra en la burbuja. */
  readonly text: string;
  /** Nombre accesible del botón (p. ej. "Más información sobre Riesgo de geolocalización"). */
  readonly label: string;
}

/**
 * Icono "i" en un círculo con una descripción emergente. Se abre al pasar el
 * ratón, al enfocarlo con el teclado y al tocarlo (en móvil no hay hover), y
 * se cierra al salir, al perder el foco, con Escape o al pulsar fuera. La
 * burbuja es un `role="tooltip"` enlazado con `aria-describedby`, así que los
 * lectores de pantalla leen la descripción aunque esté oculta; el icono SVG
 * es decorativo. Estilos en index.css (`.info-tip*`): la burbuja se ancla a la
 * fila contenedora (`.score-row`, `position: relative`), no al icono, para no
 * salirse de la pantalla en móvil.
 */
const InfoTip: React.FC<InfoTipProps> = ({ text, label }) => {
  const id = useId();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement>(null);
  // Un clic con el ratón también enfoca el botón: sin esto, onFocus y onClick
  // competirían por el mismo gesto. Solo el foco por teclado abre en onFocus.
  const pointerFocusRef = useRef(false);

  useEffect(() => {
    if (!open) return;
    const closeIfOutside = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('pointerdown', closeIfOutside);
    return () => document.removeEventListener('pointerdown', closeIfOutside);
  }, [open]);

  return (
    <span className="info-tip" ref={rootRef}>
      <button
        type="button"
        className="info-tip-button"
        aria-label={label}
        aria-describedby={id}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onPointerDown={() => {
          pointerFocusRef.current = true;
        }}
        onFocus={() => {
          if (!pointerFocusRef.current) setOpen(true);
        }}
        onBlur={() => {
          pointerFocusRef.current = false;
          setOpen(false);
        }}
        onClick={() => setOpen(true)}
        onKeyDown={(event) => {
          if (event.key === 'Escape') setOpen(false);
        }}
      >
        <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">
          <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.4" />
          <circle cx="8" cy="4.7" r="1" fill="currentColor" />
          <rect x="7.25" y="6.7" width="1.5" height="4.9" rx="0.75" fill="currentColor" />
        </svg>
      </button>
      <span id={id} role="tooltip" className={open ? 'info-tip-bubble info-tip-bubble--open' : 'info-tip-bubble'}>
        {text}
      </span>
    </span>
  );
};

export default InfoTip;
