import type { AuthStatus, ExposureReport, Platform } from './types';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:3000';

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    credentials: 'include', // imprescindible: la sesión va en cookie firmada
    ...options,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
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
};
