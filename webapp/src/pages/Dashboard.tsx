import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { api } from '../api';
import HourlyActivityChart from '../components/HourlyActivityChart';
import AiSummaryCard from '../components/AiSummaryCard';
import DownloadReportButton from '../components/DownloadReportButton';
import LanguageSwitcher from '../components/LanguageSwitcher';
import LocationMap from '../components/LocationMap';
import PopulationNarrowingTable from '../components/PopulationNarrowingTable';
import InferredAttributesList from '../components/InferredAttributesList';
import ScoreBar from '../components/ScoreBar';
import { ManualTraitsSelector } from '../components/ManualTraitsSelector';
import type { ExposureReport, Platform, ManualAttribute } from '../types';

function readPlatform(): Platform {
  const value = new URLSearchParams(window.location.search).get('platform');
  return value === 'instagram' ? 'instagram' : 'reddit';
}

// Duración del ciclo de rotación del spinner, en ms.
const SPINNER_PERIOD_MS = 800;

// Las líneas de fotos son las únicas con contador -- se muestran siempre
// igual tanto si siguen en curso ("Analizando fotos (3/10)...") como cuando
// ya terminaron ("Fotos analizadas (10/10)"), a partir de photos_analyzed/
// total_photos (ver app/vision/geolocation.py). Hay DOS líneas de fotos
// independientes -- geolocalización (DINOv2) y análisis de contenido
// (Moondream2), dos modelos y dos propósitos sobre la misma foto -- así que
// se parametriza el verbo/sustantivo (ya traducidos por el llamador) en vez
// de duplicar la función.
function formatPhotosLabel(
  counts: Record<string, unknown>,
  done: boolean,
  verb: string,
  doneLabel: string,
): string {
  const analyzed = counts.photos_analyzed;
  const total = counts.total_photos;
  if (typeof analyzed !== 'number' || typeof total !== 'number') {
    return done ? doneLabel : `${verb}...`;
  }
  return done ? `${doneLabel} (${analyzed}/${total})` : `${verb} (${analyzed}/${total})...`;
}

function StatusIcon({ done }: { done: boolean }) {
  // `key` distinta a propósito: si spinner y check compartieran nodo del DOM
  // (mismo tipo de elemento -- <span> -- en la misma posición), React lo
  // reciclaría y dejaría pegado el `transform: rotate(...)` que el efecto de
  // arriba aplica imperativamente sobre `.spinner` en cada fotograma -- el
  // check heredaría el último ángulo del giro (a veces boca abajo) en vez de
  // aparecer recto. La `key` obliga a montar un nodo nuevo y limpio, sin
  // transform heredado, aunque ambos sean <span>.
  return done ? (
    <span key="done" className="progress-icon progress-icon-done" aria-hidden="true">✓</span>
  ) : (
    <span key="spinner" className="spinner spinner-sm" aria-hidden="true" />
  );
}

// Progreso de una pista de fotos (geolocalización o análisis de contenido,
// ver docstring del estado más abajo) como "terminada": ambos contadores son
// números válidos y ya se ha llegado al total.
function isTrackDone(counts: Record<string, unknown>): boolean {
  const analyzed = counts.photos_analyzed;
  const total = counts.total_photos;
  return typeof analyzed === 'number' && typeof total === 'number' && analyzed >= total && total > 0;
}

const Dashboard: React.FC = () => {
  const { t } = useTranslation();
  const [platform] = useState<Platform>(readPlatform);
  const [report, setReport] = useState<ExposureReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isManualTraitsOpen, setIsManualTraitsOpen] = useState(false);
  // Fases YA completadas del pipeline, en el orden en que han llegado por
  // el stream -- para pintar la lista de progreso en vivo (no un
  // temporizador simulado: cada línea corresponde a un evento real emitido
  // por el backend, ver app/progress.py y analysis_router.py). Desde que
  // el backend emite CÓDIGOS de fase (ver app/stages.py) en vez de texto
  // ya renderizado en español, se guarda el código tal cual y se traduce
  // solo al pintar (t('dashboard.stages.' + code)), con el propio código
  // como fallback por si llega uno que el frontend todavía no conoce
  // (backend más nuevo que el frontend desplegado).
  const [completedStages, setCompletedStages] = useState<string[]>([]);
  const [currentStage, setCurrentStage] = useState<string | null>(null);
  // El análisis de fotos corre en PARALELO con el resto del pipeline desde
  // el principio (ver analysis_router._build_report), así que sus eventos
  // se muestran en su propia línea independiente, no mezclados con la fase
  // "general" en curso -- si no, al intercalarse en el mismo stream SSE,
  // una foto a medio analizar se marcaría por error como fase "completada"
  // cada vez que llegara un evento distinto del pipeline general. Son DOS
  // líneas, no una: geolocalización por similitud visual (DINOv2,
  // track:"geolocalizacion") y análisis de contenido -- aficiones, pareja
  // -- (Moondream2, track:"fotos", ver app/vision/geolocation.py). Avanzan
  // de forma INDEPENDIENTE entre sí (ver ADR-33): el backend emite el
  // progreso de cada pista en el momento exacto en que esa etapa termina
  // para cada foto, no en bloque -- así que es normal y esperado que una
  // pista (normalmente "geolocalizacion", DINOv2 es mucho más rápido)
  // llegue al 100% bastante antes que la otra ("fotos", Moondream2 es
  // mucho más lento), en vez de subir siempre pegadas la una a la otra.
  const [photosCounts, setPhotosCounts] = useState<Record<string, unknown> | null>(null);
  const [photosDone, setPhotosDone] = useState(false);
  const [geoCounts, setGeoCounts] = useState<Record<string, unknown> | null>(null);
  const [geoDone, setGeoDone] = useState(false);
  const navigate = useNavigate();
  const stopStreamRef = useRef<(() => void) | null>(null);
  // Ref auxiliar para poder leer la fase "actual" dentro del callback del
  // stream sin depender de un closure obsoleto de setState (evita tener
  // que meter currentStage como dependencia del useEffect y re-suscribirse
  // al stream en cada cambio de fase).
  const currentStageRef = useRef<string | null>(null);
  // Cola de líneas ya completadas pendientes de PINTAR, más el temporizador
  // que las va sacando de una en una con un mínimo de 500ms entre cada
  // aparición -- así, aunque varios eventos lleguen del backend casi a la
  // vez, el listado no salta de golpe (más legible para seguir en vivo).
  const revealQueueRef = useRef<string[]>([]);
  const revealingRef = useRef(false);
  // Contenedor de la pantalla de carga: se usa para localizar TODOS los
  // spinners que haya dentro (cabecera, fase en curso, fotos) y girarlos
  // todos a la vez -- ver el useEffect de más abajo. No se guarda una
  // referencia a cada spinner por separado porque la fase en curso y la de
  // fotos aparecen y desaparecen del DOM según avanza el análisis.
  const progressScreenRef = useRef<HTMLDivElement>(null);

  const enqueueCompleted = (stage: string) => {
    revealQueueRef.current.push(stage);
    if (revealingRef.current) return;
    revealingRef.current = true;
    const step = () => {
      const next = revealQueueRef.current.shift();
      if (next === undefined) {
        revealingRef.current = false;
        return;
      }
      setCompletedStages((prev) => [...prev, next]);
      setTimeout(step, 500);
    };
    step();
  };

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
          const counts: Record<string, unknown> = { ...event };
          delete counts.done;
          delete counts.stage;
          delete counts.track;
          const { stage } = event;

          if (event.track === 'fotos') {
            setPhotosCounts(counts);
            setPhotosDone(isTrackDone(counts));
            return;
          }

          if (event.track === 'geolocalizacion') {
            setGeoCounts(counts);
            setGeoDone(isTrackDone(counts));
            return;
          }

          const previousStage = currentStageRef.current;
          currentStageRef.current = stage;
          // Solo se marca como "completada" cuando la fase cambia de
          // verdad -- una misma fase puede repetirse varias veces seguidas
          // y en ese caso debe seguir mostrándose como la fase EN CURSO, no
          // duplicarse en la lista de completadas.
          if (previousStage && previousStage !== stage) {
            enqueueCompleted(previousStage);
          }
          setCurrentStage(stage);
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
  }, [platform, navigate]);

  // Gira TODOS los spinners de la pantalla de carga a la vez, de verdad:
  // en vez de fiarse de que el `animation-delay` de CSS los deje en fase
  // (no lo hace -- cada elemento cuenta ese desfase desde el momento en
  // que ÉL MISMO se monta en el DOM, no desde un reloj compartido, así que
  // el de cabecera, el de la fase en curso y el de fotos -- que aparecen
  // en instantes distintos -- acaban desincronizados aunque compartan el
  // mismo valor de delay), se calcula el ángulo a partir de un único
  // cronómetro (arrancado una vez, al montar el componente) y se aplica
  // como `transform: rotate(...)` a TODOS los `.spinner` presentes en cada
  // fotograma. Como todos leen el mismo reloj en el mismo instante, es
  // imposible que se desincronicen entre sí, sin importar cuándo entró
  // cada uno en el DOM.
  useEffect(() => {
    if (!loading) return;
    const start = performance.now();
    let frameId: number;

    const tick = () => {
      const elapsed = performance.now() - start;
      const angle = ((elapsed % SPINNER_PERIOD_MS) / SPINNER_PERIOD_MS) * 360;
      progressScreenRef.current?.querySelectorAll<HTMLElement>('.spinner').forEach((el) => {
        el.style.transform = `rotate(${angle}deg)`;
      });
      frameId = requestAnimationFrame(tick);
    };
    frameId = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(frameId);
  }, [loading]);

  const handleLogout = async () => {
    try {
      await api.logout(platform);
    } finally {
      navigate('/', { replace: true });
    }
  };

  const handleApplyTraits = async (traits: ManualAttribute[]) => {
    if (!report || traits.length === 0) return;
    try {
      const newReport = await api.recalculateReport({
        report,
        manual_attributes: traits,
      });
      setReport(newReport);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : t('dashboard.manualTraits.recalculateError'));
    }
  };

  const handleCancelAnalysis = async () => {
    stopStreamRef.current?.();
    try {
      await api.logout(platform);
    } finally {
      navigate('/', { replace: true });
    }
  };

  const platformLabel = platform === 'instagram' ? 'Instagram' : 'Reddit';
  const usernamePrefix = platform === 'instagram' ? '@' : 'u/';
  const groupingLabel =
    platform === 'instagram' ? t('dashboard.groupingLabelInstagram') : t('dashboard.groupingLabelReddit');
  const groupingPrefix = platform === 'instagram' ? '#' : 'r/';

  if (loading) {
    return (
      <div className="page">
        <div className="progress-screen" ref={progressScreenRef}>
          <p className="progress-heading">
            <span className="spinner" aria-hidden="true" />
            {t('dashboard.analyzing', { platform: platformLabel })}
          </p>
          <div className="progress-frame">
            <ul className="progress-list">
              {completedStages.map((stage, i) => (
                <li key={`${stage}-${i}`} className="progress-done">
                  <span className="progress-icon progress-icon-done" aria-hidden="true">✓</span>
                  {t(`dashboard.stages.${stage}`, { defaultValue: stage })}
                </li>
              ))}
              {currentStage && (
                <li className="progress-current">
                  <span className="spinner spinner-sm" aria-hidden="true" />
                  {t(`dashboard.stages.${currentStage}`, { defaultValue: currentStage })}
                </li>
              )}
              {geoCounts && (
                <li className={geoDone ? 'progress-done' : 'progress-current'}>
                  <StatusIcon done={geoDone} />
                  {formatPhotosLabel(
                    geoCounts,
                    geoDone,
                    t('dashboard.photos.geolocatingVerb'),
                    t('dashboard.photos.geolocatedDone'),
                  )}
                </li>
              )}
              {photosCounts && (
                <li className={photosDone ? 'progress-done' : 'progress-current'}>
                  <StatusIcon done={photosDone} />
                  {formatPhotosLabel(
                    photosCounts,
                    photosDone,
                    t('dashboard.photos.analyzingVerb'),
                    t('dashboard.photos.analyzedDone'),
                  )}
                </li>
              )}
            </ul>
          </div>
          <div className="progress-actions">
            <button className="btn-secondary" type="button" onClick={handleCancelAnalysis}>
              {t('dashboard.cancelAnalysis')}
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page">
        <p className="error">{t('dashboard.errorPrefix', { error })}</p>
        <button type="button" onClick={handleLogout}>{t('dashboard.backToStart')}</button>
      </div>
    );
  }

  if (!report) return null;

  // Hubo fotos analizadas (report.image_location_points no está vacío) pero
  // ninguna llegó al consenso mínimo para estimar una comunidad autónoma /
  // provincia de residencia (ver HIGH_CONFIDENCE*/MODERATE_CONFIDENCE* en
  // report/generator.py) -- se comprueba mirando si existe algún paso de
  // "Qué se puede inferir sobre ti" de categoría "ubicacion" cuyo origen
  // sea "imagen". Independiente de si el TEXTO sí dio una ubicación: este
  // aviso es específicamente sobre la fiabilidad del análisis de imagen.
  const imageLocationConfidenceInsufficient =
    report.image_location_points.length > 0 &&
    !report.population_narrowing.some((step) => step.category === 'ubicacion' && step.source === 'imagen');

  return (
    <div className="page dashboard">
      <LanguageSwitcher />
      <header className="dashboard-header">
        <h1 className="dashboard-title">
          {report.avatar_url && (
            <img
              src={report.avatar_url}
              alt={t('dashboard.avatarAlt', { username: report.username })}
              className="dashboard-avatar"
              // Las URLs de foto de perfil de Reddit/Instagram pueden llevar
              // tokens firmados con expiración, o el usuario puede no tener
              // avatar personalizado a pesar de que el campo venga presente
              // -- si la carga falla, se oculta en vez de mostrar el icono
              // roto del navegador; el título sigue siendo legible sin ella.
              onError={(e) => {
                e.currentTarget.style.display = 'none';
              }}
            />
          )}
          <span>
            {t('dashboard.title', { prefix: usernamePrefix, username: report.username })}{' '}
            <span className="platform-tag">({platformLabel})</span>
          </span>
        </h1>
        <div className="dashboard-header-actions">
          <DownloadReportButton report={report} />
          <button type="button" className="btn-secondary" onClick={handleLogout}>
            {t('dashboard.logout')}
          </button>
        </div>
      </header>

      <p className="meta">
        {t('dashboard.meta', {
          date: new Date(report.generated_at).toLocaleString(),
          count: report.n_posts_analyzed,
        })}
      </p>

      <section className="card">
        <h2>{t('dashboard.overallScore', { score: report.privacy_score.overall_score.toFixed(1) })}</h2>
        <ScoreBar
          label={t('dashboard.scoreLabels.geolocationRisk')}
          value={report.privacy_score.geolocation_risk}
          tooltip={t('dashboard.scoreLabels.geolocationRiskTooltip')}
        />
        <ScoreBar
          label={t('dashboard.scoreLabels.inferableDataRisk')}
          value={report.privacy_score.inferable_data_risk}
          tooltip={t('dashboard.scoreLabels.inferableDataRiskTooltip')}
        />
        <ScoreBar
          label={t('dashboard.scoreLabels.deanonymizationEase')}
          value={report.privacy_score.deanonymization_ease}
          tooltip={t('dashboard.scoreLabels.deanonymizationEaseTooltip')}
        />
        <p className="note">{report.privacy_score.breakdown_explanation.identity_consistency}</p>
      </section>

      <section className="card">
        <h2>{t('dashboard.whatCanBeInferred')}</h2>
        <ManualTraitsSelector 
          isOpen={isManualTraitsOpen}
          setIsOpen={setIsManualTraitsOpen}
          onApplyTraits={handleApplyTraits}
        />
        <PopulationNarrowingTable
          steps={report.population_narrowing}
          remainingPopulationAllTraits={report.remaining_population_all_traits}
          remainingPopulationAllTraitsProportion={report.remaining_population_all_traits_proportion}
        />
        <InferredAttributesList attributes={report.inferred_attributes} />
      </section>

      <section className="card">
        <h2>{t('dashboard.estimatedLocations')}</h2>
        {imageLocationConfidenceInsufficient && (
          <p className="note">{t('dashboard.imageLocationConfidenceInsufficient')}</p>
        )}
        <LocationMap points={report.image_location_points} platform={report.platform} available={report.geolocation_available} />
      </section>

      <section className="card">
        <h2>{t('dashboard.hourlyPattern')}</h2>
        <HourlyActivityChart hourlyData={report.fingerprint.avg_posts_per_hour} />
      </section>

      <section className="card">
        <h2>{t('dashboard.writingProfile')}</h2>
        <ul className="kv-list">
          <li>{t('dashboard.avgSentenceLength', { value: report.fingerprint.avg_sentence_length })}</li>
          <li>{t('dashboard.vocabularyRichness', { value: report.fingerprint.vocabulary_richness })}</li>
          <li>{t('dashboard.emojiUsage', { value: (report.fingerprint.emoji_usage_rate * 100).toFixed(2) })}</li>
          <li>{t('dashboard.detectedLanguage', { value: report.fingerprint.detected_language })}</li>
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
