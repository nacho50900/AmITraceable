import { afterEach, describe, expect, test, vi } from 'vitest';
import { AiSummaryUnavailableError, api } from '../api';
import { makeExposureReport } from './fixtures';

function mockFetchOnce(status: number, body: unknown, ok = status >= 200 && status < 300) {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
  });
}

describe('api', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test('petición exitosa devuelve el cuerpo JSON parseado', async () => {
    const fetchMock = mockFetchOnce(200, { authenticated: true });
    vi.stubGlobal('fetch', fetchMock);

    const result = await api.authStatus('reddit');

    expect(result).toEqual({ authenticated: true });
  });

  test('todas las peticiones incluyen credentials: include (sesión vía cookie)', async () => {
    const fetchMock = mockFetchOnce(200, { authenticated: false });
    vi.stubGlobal('fetch', fetchMock);

    await api.authStatus('instagram');

    const [, options] = fetchMock.mock.calls[0];
    expect(options.credentials).toBe('include');
  });

  test('error genérico (4xx/5xx que no es 503) lanza Error con el detail del backend', async () => {
    const fetchMock = mockFetchOnce(401, { detail: 'No autenticado con Reddit' }, false);
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.authStatus('reddit')).rejects.toThrow('No autenticado con Reddit');
  });

  test('error genérico sin detail en el cuerpo usa el fallback "Error {status}"', async () => {
    const fetchMock = mockFetchOnce(500, {}, false);
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.authStatus('reddit')).rejects.toThrow('Error 500');
  });

  test('error genérico con cuerpo no parseable como JSON no rompe, usa fallback', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => {
        throw new Error('body no es JSON válido');
      },
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.authStatus('reddit')).rejects.toThrow('Error 500');
  });

  test('503 lanza específicamente AiSummaryUnavailableError, no un Error genérico', async () => {
    const fetchMock = mockFetchOnce(503, { detail: 'Cuota agotada' }, false);
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.aiSummary(makeExposureReport())).rejects.toBeInstanceOf(AiSummaryUnavailableError);
  });

  test('503 sin detail en el cuerpo usa el mensaje por defecto', async () => {
    const fetchMock = mockFetchOnce(503, {}, false);
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.aiSummary(makeExposureReport())).rejects.toThrow('Análisis con IA no disponible ahora mismo.');
  });

  test('loginUrl construye la URL de login sin hacer ninguna petición', () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const url = api.loginUrl('reddit');

    expect(url).toContain('/auth/reddit/login');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test('logout hace POST a /auth/{platform}/logout', async () => {
    const fetchMock = mockFetchOnce(200, { status: 'ok' });
    vi.stubGlobal('fetch', fetchMock);

    await api.logout('instagram');

    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toContain('/auth/instagram/logout');
    expect(options.method).toBe('POST');
  });

  test('analyze hace POST a /api/analyze/{platform}', async () => {
    const fetchMock = mockFetchOnce(200, makeExposureReport());
    vi.stubGlobal('fetch', fetchMock);

    await api.analyze('reddit');

    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toContain('/api/analyze/reddit');
    expect(options.method).toBe('POST');
  });

  test('aiSummary hace POST a /api/analyze/ai-summary con el informe como body JSON', async () => {
    const fetchMock = mockFetchOnce(200, { conclusions: ['x'] });
    vi.stubGlobal('fetch', fetchMock);
    const report = makeExposureReport();

    await api.aiSummary(report);

    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toContain('/api/analyze/ai-summary');
    expect(options.method).toBe('POST');
    expect(options.headers['Content-Type']).toBe('application/json');
    expect(JSON.parse(options.body)).toEqual(report);
  });
});
