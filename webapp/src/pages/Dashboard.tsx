import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import HourlyActivityChart from '../components/HourlyActivityChart';
import ScoreBar from '../components/ScoreBar';
import type { ExposureReport } from '../types';

const Dashboard: React.FC = () => {
  const [report, setReport] = useState<ExposureReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    api
      .authStatus()
      .then((status) => {
        if (!status.authenticated) {
          navigate('/');
          return undefined;
        }
        return api.analyze();
      })
      .then((data) => data && setReport(data))
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [navigate]);

  const handleLogout = async () => {
    await api.logout();
    navigate('/');
  };

  if (loading) {
    return (
      <div className="page">
        <p>Analizando tu actividad pública en Reddit… esto puede tardar unos segundos.</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page">
        <p className="error">No se pudo generar el informe: {error}</p>
        <button onClick={handleLogout}>Volver al inicio</button>
      </div>
    );
  }

  if (!report) return null;

  return (
    <div className="page dashboard">
      <header className="dashboard-header">
        <h1>Informe de exposición de u/{report.username}</h1>
        <button className="btn-secondary" onClick={handleLogout}>
          Cerrar sesión y borrar datos
        </button>
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
        {report.inferred_attributes.length === 0 ? (
          <p>No se ha detectado ningún atributo personal claro con las heurísticas actuales.</p>
        ) : (
          <ul className="attributes-list">
            {report.inferred_attributes.map((attr, i) => (
              <li key={i}>
                <strong>{attr.category}:</strong> {attr.value}{' '}
                <span className="confidence">(confianza {(attr.confidence * 100).toFixed(0)}%)</span>
                <div className="evidence">
                  {attr.evidence.slice(0, 3).map((link, j) => (
                    <a key={j} href={link} target="_blank" rel="noreferrer">
                      Evidencia {j + 1}
                    </a>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        )}
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
        <h3>Subreddits más frecuentes</h3>
        <p>{report.fingerprint.top_subreddits.map(([s, c]) => `r/${s} (${c})`).join(', ')}</p>
      </section>

      <section className="card recommendations">
        <h2>Recomendaciones</h2>
        <ul>
          {report.recommendations.map((rec, i) => (
            <li key={i}>{rec}</li>
          ))}
        </ul>
      </section>
    </div>
  );
};

export default Dashboard;
