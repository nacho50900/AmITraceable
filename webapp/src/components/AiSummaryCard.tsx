import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AiSummaryUnavailableError, api } from '../api';
import type { ExposureReport } from '../types';

interface AiSummaryCardProps {
  report: ExposureReport;
}

type Status = 'loading' | 'success' | 'empty' | 'unavailable' | 'error';

const AiSummaryCard: React.FC<AiSummaryCardProps> = ({ report }) => {
  const { t } = useTranslation();
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
        setMessage(err instanceof Error ? err.message : t('components.aiSummary.unexpectedError'));
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
      <h2>{t('components.aiSummary.title')}</h2>

      {status === 'loading' && <p className="note">{t('components.aiSummary.loading')}</p>}

      {status === 'success' && (
        <>
          {/* verdict y conclusions vienen del backend (Mistral) ya
              generados en un idioma fijo -- traducir contenido generado
              por IA queda fuera del alcance de esta primera fase de i18n
              (solo UI estática). */}
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

      {status === 'empty' && <p className="note">{t('components.aiSummary.empty')}</p>}

      {status === 'unavailable' && <p className="note">{message}</p>}

      {status === 'error' && (
        <p className="note error-text">
          {t('components.aiSummary.errorPrefix', { message })}
          <br />
          <button type="button" className="btn-secondary" onClick={runAnalysis}>
            {t('components.aiSummary.retry')}
          </button>
        </p>
      )}
    </section>
  );
};

export default AiSummaryCard;
