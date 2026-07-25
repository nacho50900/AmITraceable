import type { AnalysisProgressEvent, AuthStatus, ExposureReport, Platform } from './types';

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
      throw new AiSummaryUnavailableError(body.detail ?? 'Análisis con IA no disponible ahora mismo.');
    }
    throw new Error(body.detail ?? `Error ${res.status}`);
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

    source.onmessage = (message) => {
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
      // distinguir ambos casos y no duplicar el error.
      if (source.readyState !== EventSource.CLOSED) {
        onEvent({ done: true, error: 'Se perdió la conexión con el servidor durante el análisis.' });
      }
      source.close();
    };

    return () => source.close();
  },
  // Endpoint aislado del pipeline principal: manda el informe YA generado
  // (que el frontend ya tiene en memoria) para que una IA externa (Mistral,
  // tier gratuito) dé conclusiones priorizadas. Si no está disponible,
  // lanza AiSummaryUnavailableError en vez de un Error genérico.
  aiSummary: (report: ExposureReport): Promise<{ conclusions: string[] }> =>
    request<{ conclusions: string[] }>('/api/analyze/ai-summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(report),
    }),
};
