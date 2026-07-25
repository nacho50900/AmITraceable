import '@testing-library/jest-dom';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { api } from '../api';
import Dashboard from '../pages/Dashboard';
import { makeExposureReport } from './fixtures';
import type { AnalysisProgressEvent } from '../types';

const mockNavigate = vi.hoisted(() => vi.fn());

// recharts (usado por HourlyActivityChart, renderizado dentro de Dashboard)
// necesita ResizeObserver para su ResponsiveContainer; jsdom no lo implementa.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', ResizeObserverStub);

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return { ...actual, useNavigate: () => mockNavigate };
});

vi.mock('../api', async () => {
  const actual = await vi.importActual<typeof import('../api')>('../api');
  return {
    ...actual,
    api: {
      authStatus: vi.fn(),
      analyzeStream: vi.fn(),
      logout: vi.fn(),
      loginUrl: (platform: string) => `http://localhost:3000/auth/${platform}/login`,
      // AiSummaryCard se dispara solo en cuanto hay informe; en estos tests
      // no nos interesa su comportamiento, así que se deja "colgado" sin
      // resolver para que no interfiera con las aserciones del Dashboard.
      aiSummary: vi.fn(() => new Promise(() => {})),
    },
  };
});

// LocationMap usa react-leaflet, que depende de APIs de navegador real que
// jsdom no implementa de forma fiable (igual que en LocationMap.test.tsx).
// Se mockea aquí también porque Dashboard renderiza LocationMap dentro de
// su árbol completo.
vi.mock('react-leaflet', () => ({
  MapContainer: ({ children }: any) => <div data-testid="map-container">{children}</div>,
  TileLayer: () => <div />,
  CircleMarker: ({ children }: any) => <div>{children}</div>,
  Tooltip: ({ children }: any) => <div>{children}</div>,
  Popup: ({ children }: any) => <div>{children}</div>,
}));

function renderDashboard() {
  return render(
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Dashboard />
    </MemoryRouter>
  );
}

/** Helper: monta analyzeStream para que emita la secuencia de eventos dada
 * (síncronamente, uno tras otro) y devuelva una función de limpieza mock. */
function mockStream(events: AnalysisProgressEvent[]) {
  const stop = vi.fn();
  vi.mocked(api.analyzeStream).mockImplementation((_platform, onEvent) => {
    events.forEach((event) => onEvent(event));
    return stop;
  });
  return stop;
}

function neverEmits() {
  const stop = vi.fn();
  vi.mocked(api.analyzeStream).mockImplementation(() => stop);
  return stop;
}

describe('Dashboard', () => {
  beforeEach(() => {
    window.history.pushState({}, '', '/'); // sin query param -> plataforma por defecto (reddit)
    vi.mocked(api.authStatus).mockReset();
    vi.mocked(api.analyzeStream).mockReset();
    vi.mocked(api.logout).mockReset();
    mockNavigate.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('muestra el mensaje de carga mientras se resuelve el análisis', () => {
    vi.mocked(api.authStatus).mockImplementation(() => new Promise(() => {})); // nunca resuelve
    renderDashboard();

    expect(screen.getByText(/Analizando tu actividad pública en Reddit/)).toBeInTheDocument();
  });

  test('usuario no autenticado: redirige a "/" sin llegar a abrir el stream', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: false });
    renderDashboard();

    await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith('/'));
    expect(api.analyzeStream).not.toHaveBeenCalled();
  });

  test('usuario autenticado: abre el stream y muestra el informe final', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([{ done: true, report: makeExposureReport({ username: 'usuario_prueba' }) }]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/usuario_prueba/)).toBeInTheDocument();
    });
    expect(api.analyzeStream).toHaveBeenCalledWith('reddit', expect.any(Function));
  });

  test('muestra las fases de progreso en vivo mientras el análisis está en curso', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      { done: false, stage: 'Conectando con la plataforma...' },
      { done: false, stage: 'Leyendo publicaciones...' },
      { done: false, stage: 'Analizando vocabulario...' },
    ]);
    renderDashboard();

    await waitFor(() => {
      // Las dos primeras ya "completadas" (llegó una fase posterior);
      // la última es la fase actual en curso.
      expect(screen.getByText('Conectando con la plataforma...')).toBeInTheDocument();
      expect(screen.getByText('Leyendo publicaciones...')).toBeInTheDocument();
      expect(screen.getByText('Analizando vocabulario...')).toBeInTheDocument();
    });
  });

  test('error durante el análisis: muestra el mensaje y el botón de volver', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([{ done: true, error: 'fallo de red simulado' }]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/fallo de red simulado/)).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: 'Volver al inicio' })).toBeInTheDocument();
  });

  test('botón "Volver al inicio" tras un error hace logout y navega a "/"', async () => {
    const user = userEvent.setup();
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([{ done: true, error: 'fallo' }]);
    vi.mocked(api.logout).mockResolvedValue({ status: 'ok' });
    renderDashboard();

    await screen.findByRole('button', { name: 'Volver al inicio' });
    await user.click(screen.getByRole('button', { name: 'Volver al inicio' }));

    await waitFor(() => {
      expect(api.logout).toHaveBeenCalledWith('reddit');
      expect(mockNavigate).toHaveBeenCalledWith('/');
    });
  });

  test('botón "Cerrar sesión y borrar datos" hace logout y navega a "/"', async () => {
    const user = userEvent.setup();
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([{ done: true, report: makeExposureReport() }]);
    vi.mocked(api.logout).mockResolvedValue({ status: 'ok' });
    renderDashboard();

    await screen.findByRole('button', { name: 'Cerrar sesión y borrar datos' });
    await user.click(screen.getByRole('button', { name: 'Cerrar sesión y borrar datos' }));

    await waitFor(() => {
      expect(api.logout).toHaveBeenCalledWith('reddit');
      expect(mockNavigate).toHaveBeenCalledWith('/');
    });
  });

  test('se desmonta durante la carga: cierra el stream (función de limpieza)', () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    const stop = neverEmits();
    const { unmount } = renderDashboard();

    unmount();

    // authStatus resuelve async, así que el stream puede no haberse abierto
    // aún en el momento del unmount -- lo importante es que, si se abrió,
    // stop() haya sido invocada y no quede una conexión colgada.
    return waitFor(() => {
      if (vi.mocked(api.analyzeStream).mock.calls.length > 0) {
        expect(stop).toHaveBeenCalled();
      }
    });
  });

  test('plataforma reddit (por defecto): usa prefijo "u/" y etiqueta de subreddits', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      {
        done: true,
        report: makeExposureReport({
          platform: 'reddit',
          username: 'pepito',
          fingerprint: {
            ...makeExposureReport().fingerprint,
            top_groups: [['madrid', 3]],
          },
        }),
      },
    ]);
    renderDashboard();

    await screen.findByText(/u\/pepito/);
    expect(screen.getByText('Subreddits más frecuentes')).toBeInTheDocument();
    expect(screen.getByText(/r\/madrid \(3\)/)).toBeInTheDocument();
  });

  test('plataforma instagram (por query param): usa prefijo "@" y etiqueta de hashtags', async () => {
    window.history.pushState({}, '', '/?platform=instagram');
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      {
        done: true,
        report: makeExposureReport({
          platform: 'instagram',
          username: 'pepita',
          fingerprint: {
            ...makeExposureReport().fingerprint,
            top_groups: [['viajes', 5]],
          },
        }),
      },
    ]);
    renderDashboard();

    await screen.findByText(/@pepita/);
    expect(screen.getByText('Hashtags más frecuentes')).toBeInTheDocument();
    expect(screen.getByText(/#viajes \(5\)/)).toBeInTheDocument();
    expect(api.analyzeStream).toHaveBeenCalledWith('instagram', expect.any(Function));
  });

  test('sin atributos inferidos: muestra el mensaje de fallback', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([{ done: true, report: makeExposureReport({ inferred_attributes: [] }) }]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/No se ha detectado ningún atributo personal claro/)).toBeInTheDocument();
    });
  });

  test('con atributos inferidos: muestra categoría, valor, confianza y hasta 3 evidencias', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      {
        done: true,
        report: makeExposureReport({
          inferred_attributes: [
            {
              category: 'ubicacion',
              value: 'Posible vínculo con Madrid',
              confidence: 0.75,
              evidence: ['https://x/1', 'https://x/2', 'https://x/3', 'https://x/4'],
            },
          ],
        }),
      },
    ]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('ubicacion:')).toBeInTheDocument();
    });
    expect(screen.getByText(/Posible vínculo con Madrid/)).toBeInTheDocument();
    expect(screen.getByText(/confianza 75%/)).toBeInTheDocument();
    // Máximo 3 enlaces de evidencia aunque haya 4 en los datos
    expect(screen.getAllByText(/Evidencia \d/)).toHaveLength(3);
  });

  test('muestra el score global redondeado a un decimal', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      {
        done: true,
        report: makeExposureReport({ privacy_score: { ...makeExposureReport().privacy_score, overall_score: 42.567 } }),
      },
    ]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/42\.6 \/ 100/)).toBeInTheDocument();
    });
  });

  test('muestra el número de publicaciones analizadas', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([{ done: true, report: makeExposureReport({ n_posts_analyzed: 77 }) }]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/77 publicaciones\/comentarios analizados/)).toBeInTheDocument();
    });
  });
});
