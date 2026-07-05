import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';

const Landing: React.FC = () => {
  const [authError] = useState(() => new URLSearchParams(window.location.search).get('auth_error'));
  const [checking, setChecking] = useState(() => authError === null);
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

  return (
    <div className="page landing">
      <h1>¿Cuánto se puede inferir de tu actividad pública?</h1>
      <p className="subtitle">
        Herramienta educativa y defensiva de análisis de exposición de identidad digital (TFG).
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
        <div className="platform-choice">
          <a className="btn-primary" href={api.loginUrl('reddit')}>
            Conectar con Reddit
          </a>
          <a className="btn-primary btn-instagram" href={api.loginUrl('instagram')}>
            Conectar con Instagram
          </a>
          <p className="note">
            Instagram requiere una cuenta profesional (Business o Creator) y que tu cuenta esté
            añadida como tester de la app mientras esté en modo desarrollo.
          </p>
        </div>
      )}
    </div>
  );
};

export default Landing;
