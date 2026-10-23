import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AiSummaryUnavailableError, api } from '../api';
import type { ExposureReport } from '../types';

interface AiSummaryCardProps {
  readonly report: ExposureReport;
}

type Status = 'loading' | 'success' | 'empty' | 'unavailable' | 'error';

const AiSummaryCard: React.FC<AiSummaryCardProps> = ({ report }) => {
  const { t, i18n } = useTranslation();
  const [status, setStatus] = useState<Status>('loading');
  const [verdict, setVerdict] = useState<string>('');
  const [conclusions, setConclusions] = useState<string[]>([]);
  const [message, setMessage] = useState<string>('');

  // Identificador de la última petición lanzada: una respuesta que llega
  // cuando ya se lanzó otra más reciente se ignora (ver `runAnalysis`).
  const latestRequestRef = useRef(0);
  // Siempre el informe más reciente, para que `runAnalysis` no dependa de
  // la identidad del objeto (ver el comentario del `useEffect` de abajo).
  const reportRef = useRef(report);
  reportRef.current = report;

  const runAnalysis = async () => {
    const requestId = ++latestRequestRef.current;
    setStatus('loading');
    try {
      const result = await api.aiSummary(reportRef.current);
      if (requestId !== latestRequestRef.current) return;
      if (!result.verdict && result.conclusions.length === 0) {
        setStatus('empty');
      } else {
        setVerdict(result.verdict);
        setConclusions(result.conclusions);
        setStatus('success');
      }
    } catch (err) {
      if (requestId !== latestRequestRef.current) return;
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
  //
  // La dependencia es una CLAVE de contenido, no el objeto `report`: el
  // Dashboard recrea ese objeto cada vez que termina la traducción de las
  // descripciones de las fotos (ver `setReport` en pages/Dashboard.tsx), y
  // con `[report]` eso lanzaba una SEGUNDA petición mientras la primera
  // seguía ocupando el modelo local. La segunda esperaba como máximo 25s al
  // lock del modelo (ver `_AI_SUMMARY_LOCK_TIMEOUT_SECONDS` en
  // backend/app/ai_analysis.py), fallaba con "modelo ocupado" y pisaba el
  // estado de la tarjeta. `generated_at` cambia en cada informe NUEVO (un
  // análisis o un recálculo), no con la traducción; el idioma se incluye
  // porque el veredicto se genera directamente en el idioma de la interfaz.
  const analysisKey = `${report.platform}|${report.username}|${report.generated_at}|${i18n.language?.split('-')[0] ?? 'es'}`;
  useEffect(() => {
    void runAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisKey]);

  return (
    <section className="card ai-summary-card">
      <h2>{t('components.aiSummary.title')}</h2>

      {status === 'loading' && <p className="note">{t('components.aiSummary.loading')}</p>}

      {status === 'success' && (
        <>
          {/* verdict y conclusions vienen del backend (modelo de IA local) ya
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

      {status === 'unavailable' && (
        <p className="note">
          {message}
          <br />
          <button type="button" className="btn-secondary" onClick={runAnalysis}>
            {t('components.aiSummary.retry')}
          </button>
        </p>
      )}

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
