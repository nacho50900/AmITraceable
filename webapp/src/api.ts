import type { AuthStatus, ExposureReport } from './types';

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

export const api = {
  authStatus: (): Promise<AuthStatus> => request<AuthStatus>('/auth/reddit/status'),
  loginUrl: (): string => `${API_URL}/auth/reddit/login`,
  logout: (): Promise<{ status: string }> =>
    request<{ status: string }>('/auth/reddit/logout', { method: 'POST' }),
  analyze: (): Promise<ExposureReport> => request<ExposureReport>('/api/analyze', { method: 'POST' }),
};
