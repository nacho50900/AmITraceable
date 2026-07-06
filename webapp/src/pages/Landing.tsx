import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { SiInstagram, SiReddit, SiX } from 'react-icons/si';
import { api } from '../api';
import type { Platform } from '../types';

interface PlatformCardData {
  // `platform` solo se rellena para plataformas realmente conectadas al
  // backend. Las tarjetas "Coming Soon" no tienen backend detrás, así que
  // se dejan sin `platform` y el botón de conexión se deshabilita solo.
  platform?: Platform;
  name: string;
  description: string;
  icon: React.ReactNode;
  cardClassName: string;
  comingSoon?: boolean;
}

// El orden de este array es el orden del mazo. Añadir una plataforma nueva
// de verdad (cuando X tenga API disponible, por ejemplo) es rellenar su
// `platform` y quitar `comingSoon`; el resto del componente ya funciona
// igual para cualquier número de cartas.
const PLATFORM_CARDS: PlatformCardData[] = [
  {
    platform: 'reddit',
    name: 'Reddit',
    description: 'Analiza tus posts y comentarios públicos.',
    icon: <SiReddit aria-hidden="true" />,
    cardClassName: 'platform-card--reddit',
  },
  {
    platform: 'instagram',
    name: 'Instagram',
    description: 'Requiere cuenta Business o Creator y estar añadido como tester de la app.',
    icon: <SiInstagram aria-hidden="true" />,
    cardClassName: 'platform-card--instagram',
  },
  {
    name: 'X',
    description: 'Pendiente de aprobación de acceso a la API. Próximamente.',
    icon: <SiX aria-hidden="true" />,
    cardClassName: 'platform-card--x',
    comingSoon: true,
  },
];

/** Distancia con signo más corta entre `index` y `activeIndex`, teniendo en
 * cuenta el "dar la vuelta" del mazo (para que funcione bien con cualquier
 * número de cartas). */
function relativeOffset(index: number, activeIndex: number, length: number): number {
  let diff = index - activeIndex;
  if (diff > length / 2) diff -= length;
  if (diff < -length / 2) diff += length;
  return diff;
}

/** Estilo de cada carta según su distancia (offset) a la carta activa.
 * offset 0 = carta activa (centrada, delante). ±1 = asoma detrás en
 * diagonal. Más lejos = oculta. */
function cardStyle(offset: number): React.CSSProperties {
  const abs = Math.abs(offset);

  if (abs === 0) {
    return { transform: 'translateX(-50%) rotate(0deg) scale(1)', zIndex: 3, opacity: 1 };
  }

  if (abs === 1) {
    const dir = offset > 0 ? 1 : -1;
    return {
      transform: `translateX(calc(-50% + ${dir * 46}px)) translateY(10px) rotate(${dir * 9}deg) scale(0.93)`,
      zIndex: 2,
      opacity: 0.85,
    };
  }

  return { transform: 'translateX(-50%) scale(0.8)', zIndex: 1, opacity: 0, pointerEvents: 'none' };
}

const SWIPE_THRESHOLD_PX = 40;

const Landing: React.FC = () => {
  const [authError] = useState(() => new URLSearchParams(window.location.search).get('auth_error'));
  const [checking, setChecking] = useState(() => authError === null);
  const [activeIndex, setActiveIndex] = useState(0);
  const dragState = useRef({ startX: 0, dragged: false });
  const navigate = useNavigate();

  useEffect(() => {
    if (authError) {
      // El usuario denegó el consentimiento (Reddit o Instagram); no
      // sabemos cuál de las dos sin más contexto, así que solo dejamos de
      // mostrar el spinner inicial (el estado `checking` ya es `false`).
      return;
    }

    // Comprobamos ambas plataformas; si cualquiera ya está autenticada,
    // saltamos directamente al dashboard de esa plataforma.
    Promise.all([api.authStatus('reddit'), api.authStatus('instagram')])
      .then(([redditStatus, instagramStatus]) => {
        if (redditStatus.authenticated) {
          navigate('/dashboard?platform=reddit');
        } else if (instagramStatus.authenticated) {
          navigate('/dashboard?platform=instagram');
        }
      })
      .finally(() => setChecking(false));
  }, [authError, navigate]);

  const goTo = (index: number) => {
    const length = PLATFORM_CARDS.length;
    setActiveIndex(((index % length) + length) % length);
  };

  // Arrastre (dedo o ratón, Pointer Events cubre ambos) sobre el mazo: si
  // se supera el umbral, avanza o retrocede una carta. No es un seguimiento
  // 1:1 del dedo (el efecto de baraja con rotación no se presta bien a
  // interpolar continuamente); es un gesto de "deslizar para pasar página".
  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    dragState.current = { startX: event.clientX, dragged: false };
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (Math.abs(event.clientX - dragState.current.startX) > 5) {
      dragState.current.dragged = true;
    }
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const delta = event.clientX - dragState.current.startX;
    if (Math.abs(delta) > SWIPE_THRESHOLD_PX) {
      goTo(activeIndex + (delta < 0 ? 1 : -1));
    }
  };

  const handleCardClick = (index: number) => {
    // Si el puntero se movió más de unos pocos píxeles fue un arrastre, no
    // un tap de selección; el arrastre ya se gestionó en handlePointerUp.
    if (dragState.current.dragged) return;
    goTo(index);
  };

  const activeCard = PLATFORM_CARDS[activeIndex];

  return (
    <div className="page landing">
      <h1 className="brand-title">AmITraceable</h1>
      <p className="subtitle">
        ¿Cuánto se puede inferir de tu actividad pública? Herramienta educativa y defensiva de
        análisis de exposición de identidad digital (TFG).
      </p>

      <div className="consent-box">
        <h2>Antes de continuar</h2>
        <ul>
          <li>Solo se analiza tu propia cuenta, nunca la de terceros.</li>
          <li>Autorizas el acceso vía OAuth oficial de la plataforma, con permisos de solo lectura.</li>
          <li>No se guarda nada: el análisis ocurre en memoria y desaparece al cerrar sesión.</li>
          <li>Puedes revocar el acceso en cualquier momento desde tu cuenta.</li>
        </ul>
      </div>

      {!checking && (
        <div className="platform-picker">
          <p className="platform-picker-hint">Desliza o elige una plataforma</p>

          <div className="deck-row">
            <button type="button" className="carousel-arrow" aria-label="Plataforma anterior" onClick={() => goTo(activeIndex - 1)}>
              ‹
            </button>

            <div
              className="card-deck"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
            >
              {PLATFORM_CARDS.map((card, index) => {
                const offset = relativeOffset(index, activeIndex, PLATFORM_CARDS.length);
                const isActive = offset === 0;
                return (
                  <div
                    key={card.name}
                    className={`platform-card ${card.cardClassName} ${isActive ? 'platform-card--active' : ''}`}
                    style={cardStyle(offset)}
                    onClick={() => !isActive && handleCardClick(index)}
                    role={isActive ? undefined : 'button'}
                    aria-label={isActive ? undefined : `Seleccionar ${card.name}`}
                    tabIndex={isActive ? -1 : 0}
                  >
                    {card.comingSoon && <span className="coming-soon-badge">Coming Soon</span>}
                    <span className="platform-card-icon">{card.icon}</span>
                    <span className="platform-card-name">{card.name}</span>
                    <span className="platform-card-description">{card.description}</span>
                  </div>
                );
              })}
            </div>

            <button type="button" className="carousel-arrow" aria-label="Siguiente plataforma" onClick={() => goTo(activeIndex + 1)}>
              ›
            </button>
          </div>

          <div className="carousel-dots">
            {PLATFORM_CARDS.map((card, index) => (
              <button
                key={card.name}
                type="button"
                aria-label={`Ir a la tarjeta de ${card.name}`}
                className={`carousel-dot ${index === activeIndex ? 'carousel-dot--active' : ''}`}
                onClick={() => goTo(index)}
              />
            ))}
          </div>

          {activeCard.comingSoon || !activeCard.platform ? (
            <button type="button" className="btn-primary deck-cta deck-cta--disabled" disabled>
              Próximamente
            </button>
          ) : (
            <a className="btn-primary deck-cta" href={api.loginUrl(activeCard.platform)}>
              Conectar con {activeCard.name} →
            </a>
          )}
        </div>
      )}
    </div>
  );
};

export default Landing;
