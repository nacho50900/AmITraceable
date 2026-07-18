import React, { useState } from 'react';
import { AiSummaryUnavailableError, api } from '../api';
import type { ExposureReport } from '../types';

interface AiSummaryCardProps {
  report: ExposureReport;
}

type Status = 'idle' | 'loading' | 'success' | 'unavailable' | 'error';

const AiSummaryCard: React.FC<AiSummaryCardProps> = ({ report }) => {
  const [status, setStatus] = useState<Status>('idle');
  const [conclusions, setConclusions] = useState<string[]>([]);
  const [message, setMessage] = useState<string>('');

  const handleAnalyze = async () => {
    setStatus('loading');
    try {
      const result = await api.aiSummary(report);
      setConclusions(result.conclusions);
      setStatus('success');
    } catch (err) {
      if (err instanceof AiSummaryUnavailableError) {
        setMessage(err.message);
        setStatus('unavailable');
      } else {
        setMessage(err instanceof Error ? err.message : 'Error inesperado.');
        setStatus('error');
      }
    }
  };

  return (
    <section className="card ai-summary-card">
      <h2>Conclusiones generadas por IA</h2>

      {status === 'idle' && (
        <>
          <p className="note">
            Un modelo de IA (Mistral AI, con sede en la UE) puede leer este informe y darte
            conclusiones priorizadas en lenguaje natural. Es opcional: se envía tu informe ya
            generado, no tus publicaciones originales.
          </p>
          <button className="btn-secondary" onClick={handleAnalyze}>
            Analizar con IA
          </button>
        </>
      )}

      {status === 'loading' && <p className="note">Generando conclusiones...</p>}

      {status === 'success' && (
        <ul className="ai-conclusions-list">
          {conclusions.map((conclusion, i) => (
            <li key={i}>{conclusion}</li>
          ))}
        </ul>
      )}

      {status === 'unavailable' && (
        <p className="note">
          {message} El resto de tu informe sigue disponible con normalidad; esta es solo una
          función complementaria opcional.
        </p>
      )}

      {status === 'error' && (
        <p className="note error-text">
          No se ha podido completar el análisis con IA ({message}). Puedes intentarlo de nuevo.
          <br />
          <button className="btn-secondary" onClick={handleAnalyze}>
            Reintentar
          </button>
        </p>
      )}
    </section>
  );
};

export default AiSummaryCard;
