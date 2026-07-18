import type { AuthStatus, ExposureReport, Platform } from './types';

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
