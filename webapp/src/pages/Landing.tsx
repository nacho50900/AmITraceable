import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';

const Landing: React.FC = () => {
  const [authError] = useState(() => new URLSearchParams(window.location.search).get('auth_error'));
  const [checking, setChecking] = useState(() => authError === null);
  const navigate = useNavigate();

  useEffect(() => {
    if (authError) {
      // El usuario denegó el consentimiento en Reddit; no hacemos nada más
      // (el estado inicial de `checking` ya se calculó como `false` arriba).
      return;
    }

    api
      .authStatus()
      .then((status) => {
        if (status.authenticated) navigate('/dashboard');
      })
      .finally(() => setChecking(false));
  }, [authError, navigate]);

  return (
    <div className="page landing">
      <h1>¿Cuánto se puede inferir de tu actividad pública?</h1>
      <p className="subtitle">
        Herramienta educativa y defensiva de análisis de exposición de identidad digital
        (TFG — versión MVP centrada en Reddit).
      </p>

      <div className="consent-box">
        <h2>Antes de continuar</h2>
        <ul>
          <li>Solo se analiza tu propia cuenta de Reddit, nunca la de terceros.</li>
          <li>Autorizas el acceso vía OAuth oficial de Reddit, con permisos de solo lectura.</li>
          <li>No se guarda nada: el análisis ocurre en memoria y desaparece al cerrar sesión.</li>
          <li>Puedes revocar el acceso en cualquier momento desde tu cuenta de Reddit.</li>
        </ul>
      </div>

      {!checking && (
        <a className="btn-primary" href={api.loginUrl()}>
          Conectar con Reddit y empezar
        </a>
      )}
    </div>
  );
};

export default Landing;
