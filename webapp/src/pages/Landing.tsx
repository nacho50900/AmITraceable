import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { SiInstagram, SiReddit } from 'react-icons/si';
import { api } from '../api';
import type { Platform } from '../types';

interface PlatformCardData {
  platform: Platform;
  name: string;
  description: string;
  icon: React.ReactNode;
  cardClassName: string;
}

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
];

// Umbral de arrastre (px): por debajo de esto, un click en la tarjeta se
// trata como click de verdad (navega); por encima, se trata como arrastre
// (no navega). Sin esto, arrastrar con el ratón activaría el enlace sin
// querer al soltar.
const DRAG_THRESHOLD_PX = 6;

const Landing: React.FC = () => {
  const [authError] = useState(() => new URLSearchParams(window.location.search).get('auth_error'));
  const [checking, setChecking] = useState(() => authError === null);
  const [activeIndex, setActiveIndex] = useState(0);
  const trackRef = useRef<HTMLDivElement>(null);
  const dragState = useRef({ dragging: false, startX: 0, startScrollLeft: 0, moved: 0 });
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

  // Detecta qué tarjeta está centrada en el carrusel mientras el usuario
  // desliza, para resaltar el indicador de puntos correspondiente.
  useEffect(() => {
    const track = trackRef.current;
    if (!track) return;

    const handleScroll = () => {
      const cardWidth = track.scrollWidth / PLATFORM_CARDS.length;
      const index = Math.round(track.scrollLeft / cardWidth);
      setActiveIndex(Math.min(Math.max(index, 0), PLATFORM_CARDS.length - 1));
    };

    track.addEventListener('scroll', handleScroll, { passive: true });
    return () => track.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToCard = (index: number) => {
    const track = trackRef.current;
    if (!track) return;
    const clamped = Math.min(Math.max(index, 0), PLATFORM_CARDS.length - 1);
    const cardWidth = track.scrollWidth / PLATFORM_CARDS.length;
    track.scrollTo({ left: cardWidth * clamped, behavior: 'smooth' });
  };

  // Arrastre con ratón/trackpad para escritorio (el scroll-snap nativo por
  // sí solo solo responde bien al dedo en móvil o a gestos de dos dedos).
  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    const track = trackRef.current;
    if (!track) return;
    dragState.current = { dragging: true, startX: event.clientX, startScrollLeft: track.scrollLeft, moved: 0 };
    track.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const track = trackRef.current;
    if (!track || !dragState.current.dragging) return;
    const delta = event.clientX - dragState.current.startX;
    dragState.current.moved = Math.max(dragState.current.moved, Math.abs(delta));
    track.scrollLeft = dragState.current.startScrollLeft - delta;
  };

  const endDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const track = trackRef.current;
    if (track) track.releasePointerCapture(event.pointerId);
    dragState.current.dragging = false;
  };

  // Si el usuario arrastró más del umbral, cancelamos el click para que no
  // navegue accidentalmente al soltar el ratón sobre la tarjeta.
  const handleCardClick = (event: React.MouseEvent<HTMLAnchorElement>) => {
    if (dragState.current.moved > DRAG_THRESHOLD_PX) {
      event.preventDefault();
    }
  };

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
          <p className="platform-picker-hint">Elige una plataforma</p>

          <div className="carousel-row">
            <button
              type="button"
              className="carousel-arrow"
              aria-label="Plataforma anterior"
              disabled={activeIndex === 0}
              onClick={() => scrollToCard(activeIndex - 1)}
            >
              ‹
            </button>

            <div
              className="card-carousel"
              ref={trackRef}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={endDrag}
              onPointerLeave={endDrag}
            >
              {PLATFORM_CARDS.map((card) => (
                <a
                  key={card.platform}
                  href={api.loginUrl(card.platform)}
                  className={`platform-card ${card.cardClassName}`}
                  onClick={handleCardClick}
                  draggable={false}
                >
                  <span className="platform-card-icon">{card.icon}</span>
                  <span className="platform-card-name">{card.name}</span>
                  <span className="platform-card-description">{card.description}</span>
                  <span className="platform-card-cta">Conectar con {card.name} →</span>
                </a>
              ))}
            </div>

            <button
              type="button"
              className="carousel-arrow"
              aria-label="Siguiente plataforma"
              disabled={activeIndex === PLATFORM_CARDS.length - 1}
              onClick={() => scrollToCard(activeIndex + 1)}
            >
              ›
            </button>
          </div>

          <div className="carousel-dots">
            {PLATFORM_CARDS.map((card, index) => (
              <button
                key={card.platform}
                type="button"
                aria-label={`Ir a la tarjeta de ${card.name}`}
                className={`carousel-dot ${index === activeIndex ? 'carousel-dot--active' : ''}`}
                onClick={() => scrollToCard(index)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Landing;
