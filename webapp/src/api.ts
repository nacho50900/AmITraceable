import i18n from './i18n';
import type { AnalysisProgressEvent, AuthStatus, ExposureReport, Platform, RecalculateRequest } from './types';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:3000';

/** Se lanza específicamente cuando el análisis con IA no está disponible
 * (503: sin API key configurada, cuota del tier gratuito agotada, o error
 * del proveedor) -- para que el frontend pueda distinguirlo de un fallo
 * real de la aplicación y mostrar un mensaje adecuado, no un error genérico. */
export class AiSummaryUnavailableError extends Error {}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    credentials: 'include', // imprescindible: la sesión va en cookie firmada
    ...options,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    if (res.status === 503) {
      // body.detail viene del backend (siempre en español -- fuera del
      // alcance de esta fase de i18n); solo el fallback local se traduce.
      throw new AiSummaryUnavailableError(body.detail ?? i18n.t('api.aiUnavailable'));
    }
    throw new Error(body.detail ?? i18n.t('api.genericError', { status: res.status }));
  }

  return res.json() as Promise<T>;
}

// Todas las plataformas siguen exactamente el mismo contrato de rutas
// (/auth/{platform}/..., /api/analyze/{platform}), así que no hace falta
// ningún mapa especial por plataforma aquí.
export const api = {
  authStatus: (platform: Platform): Promise<AuthStatus> =>
    request<AuthStatus>(`/auth/${platform}/status`),
  loginUrl: (platform: Platform): string => `${API_URL}/auth/${platform}/login`,
  logout: (platform: Platform): Promise<{ status: string }> =>
    request<{ status: string }>(`/auth/${platform}/logout`, { method: 'POST' }),
  analyze: (platform: Platform): Promise<ExposureReport> =>
    request<ExposureReport>(`/api/analyze/${platform}`, { method: 'POST' }),
  // Variante con progreso en vivo, vía Server-Sent Events (GET
  // /api/analyze/{platform}/stream) -- ver docstring de analyze_stream en
  // analysis_router.py para el formato exacto de cada evento. Devuelve una
  // función de limpieza que cierra la conexión (llamarla al desmontar el
  // componente, o tras recibir un evento con done=true).
  analyzeStream: (platform: Platform, onEvent: (event: AnalysisProgressEvent) => void): (() => void) => {
    const source = new EventSource(`${API_URL}/api/analyze/${platform}/stream`, { withCredentials: true });

    // Un corte de conexión breve (wifi inestable, roaming, WSL2 renovando
    // la red...) NO debe tirar todo el análisis en curso. El EventSource
    // del navegador ya reintenta la reconexión por sí solo mientras no lo
    // cerremos nosotros -- y en el backend (ver analyze_stream en
    // analysis_router.py), el pipeline solo se cancela cuando de verdad
    // detecta al cliente desconectado, así que basta con NO cerrar la
    // conexión a la primera y darle un margen para que se recupere sola.
    // Solo si pasan STREAM_LOST_GRACE_MS sin lograrlo se da la conexión
    // por perdida de verdad y se avisa al usuario.
    const STREAM_LOST_GRACE_MS = 10_000;
    let graceTimer: ReturnType<typeof setTimeout> | null = null;

    const cancelPendingStreamLost = () => {
      if (graceTimer !== null) {
        clearTimeout(graceTimer);
        graceTimer = null;
      }
    };

    source.onopen = () => {
      // Reconexión nativa lograda: se cancela el aviso de "conexión
      // perdida" pendiente, si lo había.
      cancelPendingStreamLost();
    };

    source.onmessage = (message) => {
      cancelPendingStreamLost(); // ha llegado algo: la conexión está viva de nuevo
      let parsed: AnalysisProgressEvent;
      try {
        parsed = JSON.parse(message.data);
      } catch {
        return; // línea mal formada: no debería pasar nunca con este backend, se ignora
      }
      onEvent(parsed);
      if (parsed.done) {
        source.close();
      }
    };

    source.onerror = () => {
      // El navegador dispara este mismo evento tanto ante una caída de red
      // real como, a veces, tras el cierre normal del stream si no
      // llegamos a cerrarlo nosotros primero arriba -- readyState permite
      // distinguir ambos casos y no duplicar el error. Mientras reintenta
      // solo (readyState CONNECTING), no se toca nada más que armar el
      // aviso de gracia una única vez -- reintentos repetidos durante ese
      // margen no lo alargan.
      if (source.readyState === EventSource.CLOSED || graceTimer !== null) return;
      graceTimer = setTimeout(() => {
        graceTimer = null;
        onEvent({ done: true, error: i18n.t('api.streamLost') });
        source.close();
      }, STREAM_LOST_GRACE_MS);
    };

    return () => {
      cancelPendingStreamLost();
      source.close();
    };
  },
  // Endpoint aislado del pipeline principal: manda el informe YA generado
  // (que el frontend ya tiene en memoria) para que una IA externa (Mistral,
  // tier gratuito) dé conclusiones priorizadas. Si no está disponible,
  // lanza AiSummaryUnavailableError en vez de un Error genérico.
  aiSummary: (report: ExposureReport): Promise<{ verdict: string; conclusions: string[] }> =>
    // Se manda el idioma de UI actual (ver src/i18n) como query param, para
    // que Mistral genere el veredicto/conclusiones DIRECTAMENTE en ese
    // idioma en la misma llamada -- ver docstring de
    // `_LANGUAGE_INSTRUCTIONS` en backend/app/ai_analysis.py sobre por qué
    // no se traduce después en vez de generar directo.
    request<{ verdict: string; conclusions: string[] }>(
      `/api/analyze/ai-summary?lang=${encodeURIComponent(i18n.language?.split('-')[0] ?? 'es')}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(report),
      },
    ),
  recalculateReport: (req: RecalculateRequest): Promise<ExposureReport> =>
    request<ExposureReport>('/api/analyze/recalculate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),
};
