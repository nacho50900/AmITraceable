import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import HourlyActivityChart from '../components/HourlyActivityChart';
import AiSummaryCard from '../components/AiSummaryCard';
import DownloadReportButton from '../components/DownloadReportButton';
import LocationMap from '../components/LocationMap';
import PopulationNarrowingTable from '../components/PopulationNarrowingTable';
import ScoreBar from '../components/ScoreBar';
import type { ExposureReport, Platform } from '../types';

function readPlatform(): Platform {
  const value = new URLSearchParams(window.location.search).get('platform');
  return value === 'instagram' ? 'instagram' : 'reddit';
}

const Dashboard: React.FC = () => {
  const [platform] = useState<Platform>(readPlatform);
  const [report, setReport] = useState<ExposureReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Fases YA completadas del pipeline, en el orden en que han llegado por
  // el stream -- para pintar la lista de progreso en vivo (no un
  // temporizador simulado: cada línea corresponde a un evento real emitido
  // por el backend, ver app/progress.py y analysis_router.py).
  const [completedStages, setCompletedStages] = useState<string[]>([]);
  const [currentStage, setCurrentStage] = useState<string | null>(null);
  const navigate = useNavigate();
  const stopStreamRef = useRef<(() => void) | null>(null);
  // Ref auxiliar para poder leer la fase "actual" dentro del callback del
  // stream sin depender de un closure obsoleto de setState (evita tener
  // que meter currentStage como dependencia del useEffect y re-suscribirse
  // al stream en cada cambio de fase).
  const currentStageRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    api.authStatus(platform).then((status) => {
      if (cancelled) return;
      if (!status.authenticated) {
        navigate('/');
        return;
      }

      stopStreamRef.current = api.analyzeStream(platform, (event) => {
        if (cancelled) return;

        if (!event.done) {
          const previousStage = currentStageRef.current;
          currentStageRef.current = event.stage;
          if (previousStage) {
            setCompletedStages((prev) => [...prev, previousStage]);
          }
          setCurrentStage(event.stage);
          return;
        }

        if ('report' in event) {
          setReport(event.report);
        } else {
          setError(event.error);
        }
        setLoading(false);
      });
    });

    return () => {
      cancelled = true;
      stopStreamRef.current?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform, navigate]);

  const handleLogout = async () => {
    await api.logout(platform);
    navigate('/');
  };

  const platformLabel = platform === 'instagram' ? 'Instagram' : 'Reddit';
  const usernamePrefix = platform === 'instagram' ? '@' : 'u/';
  const groupingLabel = platform === 'instagram' ? 'Hashtags más frecuentes' : 'Subreddits más frecuentes';
  const groupingPrefix = platform === 'instagram' ? '#' : 'r/';

  if (loading) {
    return (
      <div className="page">
        <div className="progress-screen">
          <p className="progress-heading">
            <span className="spinner" aria-hidden="true" />
            Analizando tu actividad pública en {platformLabel}…
          </p>
          <ul className="progress-list">
            {completedStages.map((stage, i) => (
              <li key={`${stage}-${i}`}>{stage}</li>
            ))}
            {currentStage && <li className="progress-current">{currentStage}</li>}
          </ul>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page">
        <p className="error">No se pudo generar el informe: {error}</p>
        <button type="button" onClick={handleLogout}>Volver al inicio</button>
      </div>
    );
  }

  if (!report) return null;

  return (
    <div className="page dashboard">
      <header className="dashboard-header">
        <h1>
          Informe de exposición de {usernamePrefix}
          {report.username} <span className="platform-tag">({platformLabel})</span>
        </h1>
        <div className="dashboard-header-actions">
          <DownloadReportButton report={report} />
          <button type="button" className="btn-secondary" onClick={handleLogout}>
            Cerrar sesión y borrar datos
          </button>
        </div>
      </header>

      <p className="meta">
        Generado el {new Date(report.generated_at).toLocaleString()} · {report.n_posts_analyzed}{' '}
        publicaciones/comentarios analizados
      </p>

      <section className="card">
        <h2>Score global de exposición: {report.privacy_score.overall_score.toFixed(1)} / 100</h2>
        <ScoreBar label="Riesgo de geolocalización" value={report.privacy_score.geolocation_risk} />
        <ScoreBar label="Datos personales inferibles" value={report.privacy_score.inferable_data_risk} />
        <ScoreBar label="Facilidad de deanonimización" value={report.privacy_score.deanonymization_ease} />
        <p className="note">{report.privacy_score.breakdown_explanation.identity_consistency}</p>
      </section>

      <section className="card">
        <h2>Qué se puede inferir sobre ti</h2>
        <PopulationNarrowingTable steps={report.population_narrowing} />
      </section>

      <section className="card">
        <h2>Ubicaciones estimadas a partir de tus fotos</h2>
        <LocationMap points={report.image_location_points} platform={report.platform} available={report.geolocation_available} />
      </section>

      <section className="card">
        <h2>Patrón horario de actividad (UTC)</h2>
        <HourlyActivityChart hourlyData={report.fingerprint.avg_posts_per_hour} />
      </section>

      <section className="card">
        <h2>Perfil de escritura</h2>
        <ul className="kv-list">
          <li>Longitud media de frase: {report.fingerprint.avg_sentence_length} palabras</li>
          <li>Riqueza de vocabulario: {report.fingerprint.vocabulary_richness}</li>
          <li>Uso de emojis: {(report.fingerprint.emoji_usage_rate * 100).toFixed(2)}%</li>
          <li>Idioma detectado: {report.fingerprint.detected_language}</li>
        </ul>
        <h3>{groupingLabel}</h3>
        <p>
          {report.fingerprint.top_groups
            .map(([s, c]) => `${groupingPrefix}${s} (${c})`)
            .join(', ')}
        </p>
      </section>

      <AiSummaryCard report={report} />
    </div>
  );
};

export default Dashboard;
