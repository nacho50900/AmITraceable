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

// Duración del ciclo de rotación del spinner, en ms.
const SPINNER_PERIOD_MS = 800;

// La línea de fotos es la única con contador -- se muestra siempre igual
// tanto si sigue en curso ("Analizando fotos (3/10)...") como cuando ya
// terminó ("Fotos analizadas (10/10)"), a partir de photos_analyzed/
// total_photos (ver app/vision/geolocation.py).
function formatPhotosLabel(counts: Record<string, unknown>, done: boolean): string {
  const analyzed = counts.photos_analyzed;
  const total = counts.total_photos;
  if (typeof analyzed !== 'number' || typeof total !== 'number') {
    return done ? 'Fotos analizadas' : 'Analizando fotos...';
  }
  return done ? `Fotos analizadas (${analyzed}/${total})` : `Analizando fotos (${analyzed}/${total})...`;
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
  // El análisis de fotos corre en PARALELO con el resto del pipeline desde
  // el principio (ver analysis_router._build_report), así que sus eventos
  // (marcados con track:"fotos", ver app/vision/geolocation.py) se
  // muestran en su propia línea independiente, no mezclados con la fase
  // "general" en curso -- si no, al intercalarse en el mismo stream SSE,
  // una foto a medio analizar se marcaría por error como fase "completada"
  // cada vez que llegara un evento distinto del pipeline general.
  const [photosCounts, setPhotosCounts] = useState<Record<string, unknown> | null>(null);
  const [photosDone, setPhotosDone] = useState(false);
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
            const analyzed = counts.photos_analyzed;
            const total = counts.total_photos;
            setPhotosDone(
              typeof analyzed === 'number' && typeof total === 'number' && analyzed >= total && total > 0
            );
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
        <div className="progress-screen" ref={progressScreenRef}>
          <p className="progress-heading">
            <span className="spinner" aria-hidden="true" />
            Analizando tu actividad pública en {platformLabel}…
          </p>
          <div className="progress-frame">
            <ul className="progress-list">
              {completedStages.map((stage, i) => (
                <li key={`${stage}-${i}`} className="progress-done">
                  <span className="progress-icon progress-icon-done" aria-hidden="true">✓</span>
                  {stage}
                </li>
              ))}
              {currentStage && (
                <li className="progress-current">
                  <span className="spinner spinner-sm" aria-hidden="true" />
                  {currentStage}
                </li>
              )}
              {photosCounts && (
                <li className={photosDone ? 'progress-done' : 'progress-current'}>
                  {photosDone ? (
                    <span className="progress-icon progress-icon-done" aria-hidden="true">✓</span>
                  ) : (
                    <span className="spinner spinner-sm" aria-hidden="true" />
                  )}
                  {formatPhotosLabel(photosCounts, photosDone)}
                </li>
              )}
            </ul>
          </div>
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
        <PopulationNarrowingTable
          steps={report.population_narrowing}
          remainingPopulationAllTraits={report.remaining_population_all_traits}
          remainingPopulationAllTraitsProportion={report.remaining_population_all_traits_proportion}
        />
      </section>

      <section className="card">
        <h2>Ubicaciones estimadas a partir de tus fotos</h2>
        {imageLocationConfidenceInsufficient && (
          <p className="note">
            El índice de confianza del análisis de las imágenes no es suficiente para estimar la
            comunidad autónoma de residencia.
          </p>
        )}
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
