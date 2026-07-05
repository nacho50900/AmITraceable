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

const Landing: React.FC = () => {
  const [authError] = useState(() => new URLSearchParams(window.location.search).get('auth_error'));
  const [checking, setChecking] = useState(() => authError === null);
  const [activeIndex, setActiveIndex] = useState(0);
  const trackRef = useRef<HTMLDivElement>(null);
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
    const cardWidth = track.scrollWidth / PLATFORM_CARDS.length;
    track.scrollTo({ left: cardWidth * index, behavior: 'smooth' });
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
          <p className="platform-picker-hint">Desliza para elegir una plataforma</p>

          <div className="card-carousel" ref={trackRef}>
            {PLATFORM_CARDS.map((card) => (
              <a
                key={card.platform}
                href={api.loginUrl(card.platform)}
                className={`platform-card ${card.cardClassName}`}
              >
                <span className="platform-card-icon">{card.icon}</span>
                <span className="platform-card-name">{card.name}</span>
                <span className="platform-card-description">{card.description}</span>
                <span className="platform-card-cta">Conectar con {card.name} →</span>
              </a>
            ))}
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
