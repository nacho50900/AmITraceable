import React, { useEffect, useState } from 'react';
import { AiSummaryUnavailableError, api } from '../api';
import type { ExposureReport } from '../types';

interface AiSummaryCardProps {
  report: ExposureReport;
}

type Status = 'loading' | 'success' | 'empty' | 'unavailable' | 'error';

const AiSummaryCard: React.FC<AiSummaryCardProps> = ({ report }) => {
  const [status, setStatus] = useState<Status>('loading');
  const [verdict, setVerdict] = useState<string>('');
  const [conclusions, setConclusions] = useState<string[]>([]);
  const [message, setMessage] = useState<string>('');

  const runAnalysis = async () => {
    setStatus('loading');
    try {
      const result = await api.aiSummary(report);
      if (!result.verdict && result.conclusions.length === 0) {
        setStatus('empty');
      } else {
        setVerdict(result.verdict);
        setConclusions(result.conclusions);
        setStatus('success');
      }
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

  // Se dispara solo, sin que el usuario tenga que pulsar nada -- el
  // informe ya generado se envía en cuanto está disponible.
  useEffect(() => {
    runAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [report]);

  return (
    <section className="card ai-summary-card">
      <h2>Conclusiones generadas por IA</h2>

      {status === 'loading' && <p className="note">Analizando el informe con IA...</p>}

      {status === 'success' && (
        <>
          {verdict && <p className="ai-verdict">{verdict}</p>}
          {conclusions.length > 0 && (
            <ul className="ai-conclusions-list">
              {conclusions.map((conclusion) => (
                <li key={conclusion}>{conclusion}</li>
              ))}
            </ul>
          )}
        </>
      )}

      {status === 'empty' && (
        <p className="note">
          La IA ha revisado tu informe y no ha encontrado ninguna conclusión que merezca la
          pena destacar más allá de lo que ya ves en el resto del dashboard.
        </p>
      )}

      {status === 'unavailable' && <p className="note">{message}</p>}

      {status === 'error' && (
        <p className="note error-text">
          No se ha podido completar el análisis con IA ({message}).
          <br />
          <button type="button" className="btn-secondary" onClick={runAnalysis}>
            Reintentar
          </button>
        </p>
      )}
    </section>
  );
};

export default AiSummaryCard;
