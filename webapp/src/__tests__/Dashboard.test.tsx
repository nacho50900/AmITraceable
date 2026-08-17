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
      { done: false, stage: 'connecting' },
      { done: false, stage: 'reading_posts' },
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

  test('fase de fotos (track paralelo): contador en vivo, no se duplica, y pasa a completada al llegar al total', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      { done: false, stage: 'reading_posts' },
      { done: false, stage: 'Analizando fotos...', photos_analyzed: 1, total_photos: 3, track: 'fotos' },
      { done: false, stage: 'detecting_attributes' },
      { done: false, stage: 'Analizando fotos...', photos_analyzed: 2, total_photos: 3, track: 'fotos' },
      { done: false, stage: 'Analizando fotos...', photos_analyzed: 3, total_photos: 3, track: 'fotos' },
    ]);
    renderDashboard();

    await waitFor(() => {
      // "Leyendo publicaciones..." queda completada al llegar la siguiente
      // fase GENERAL ("Detectando atributos..."). La fase de fotos, al
      // correr en su propio track en paralelo, no interfiere con esa
      // transición -- y al llegar a 3/3 pasa de "en curso" a "completada"
      // en su propia línea, sin duplicarse.
      expect(screen.getByText('Leyendo publicaciones...')).toBeInTheDocument();
      expect(screen.getByText('Fotos analizadas (3/3)')).toBeInTheDocument();
      expect(screen.queryAllByText(/fotos/i)).toHaveLength(1);
    });
  });

  test('fase de geolocalización (track distinto de "fotos"): su propia línea, con su propio contador', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      { done: false, stage: 'reading_posts' },
      {
        done: false,
        stage: 'Geolocalizando fotos...',
        photos_analyzed: 1,
        total_photos: 3,
        track: 'geolocalizacion',
      },
      {
        done: false,
        stage: 'Geolocalizando fotos...',
        photos_analyzed: 3,
        total_photos: 3,
        track: 'geolocalizacion',
      },
    ]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Fotos geolocalizadas (3/3)')).toBeInTheDocument();
    });
  });

  test('geolocalización (DINOv2) y análisis de contenido (Moondream2) se muestran como DOS líneas independientes, sin mezclarse', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      {
        done: false,
        stage: 'Geolocalizando fotos...',
        photos_analyzed: 2,
        total_photos: 5,
        track: 'geolocalizacion',
      },
      { done: false, stage: 'Analizando fotos...', photos_analyzed: 2, total_photos: 5, track: 'fotos' },
    ]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Geolocalizando fotos (2/5)...')).toBeInTheDocument();
      expect(screen.getByText('Analizando fotos (2/5)...')).toBeInTheDocument();
    });
  });

  test('todos los spinners visibles giran sincronizados (mismo transform en todo momento)', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      { done: false, stage: 'reading_posts' },
      { done: false, stage: 'Analizando fotos...', photos_analyzed: 1, total_photos: 3, track: 'fotos' },
    ]);
    const { container } = renderDashboard();

    // Esperar a que la fase en curso y la línea de fotos hayan aparecido
    // (se montan en instantes DISTINTOS al spinner de cabecera, que es
    // justo el escenario que antes se desincronizaba).
    await waitFor(() => {
      expect(container.querySelectorAll('.spinner')).toHaveLength(3);
    });

    // Dar tiempo a que el requestAnimationFrame compartido pinte al menos
    // un par de fotogramas.
    await new Promise((resolve) => setTimeout(resolve, 100));

    const transforms = Array.from(container.querySelectorAll<HTMLElement>('.spinner')).map(
      (el) => el.style.transform,
    );
    expect(transforms).toHaveLength(3);
    // Los tres deben tener EXACTAMENTE el mismo ángulo en este instante --
    // si estuvieran desincronizados (cada uno con su propio desfase),
    // estos valores diferirían.
    expect(new Set(transforms).size).toBe(1);
    expect(transforms[0]).toMatch(/^rotate\(/);
  });

  test('el listado revela las fases completadas de una en una, no todas de golpe', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      { done: false, stage: 'connecting' },
      { done: false, stage: 'reading_posts' },
      { done: false, stage: 'analyzing_writing_style' },
    ]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Conectando con la plataforma...')).toBeInTheDocument();
    });
    // Justo cuando aparece la primera fase completada, la siguiente
    // TODAVÍA no debe estar en pantalla -- se revelan de una en una, con
    // un hueco mínimo entre cada aparición (≥200ms), no todas a la vez
    // aunque el backend las haya emitido en ráfaga.
    expect(screen.queryByText('Leyendo publicaciones...')).not.toBeInTheDocument();

    await waitFor(
      () => {
        expect(screen.getByText('Leyendo publicaciones...')).toBeInTheDocument();
      },
      { timeout: 1000 }
    );
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

  test('muestra la foto de perfil en el título cuando el informe trae avatar_url', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      { done: true, report: makeExposureReport({ username: 'usuario_prueba', avatar_url: 'https://cdn.fake/avatar.jpg' }) },
    ]);
    renderDashboard();

    await waitFor(() => {
      const avatar = screen.getByAltText('Foto de perfil de usuario_prueba');
      expect(avatar).toBeInTheDocument();
      expect(avatar).toHaveAttribute('src', 'https://cdn.fake/avatar.jpg');
    });
  });

  test('no muestra ninguna imagen en el título cuando avatar_url es null', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([{ done: true, report: makeExposureReport({ username: 'usuario_prueba', avatar_url: null }) }]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/usuario_prueba/)).toBeInTheDocument();
    });
    expect(screen.queryByRole('img', { name: /Foto de perfil/ })).not.toBeInTheDocument();
  });

  test('la sección de población usa el título "Qué se puede inferir sobre ti", sin una segunda sección redundante', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([{ done: true, report: makeExposureReport() }]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getAllByText('Qué se puede inferir sobre ti')).toHaveLength(1);
    });
  });

  test('avisa de confianza insuficiente cuando hay fotos pero ninguna dio una ubicación de residencia fiable', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    // Fixture por defecto: 1 foto en image_location_points, pero
    // population_narrowing no trae ningún paso "ubicacion" con source "imagen".
    mockStream([{ done: true, report: makeExposureReport() }]);
    renderDashboard();

    await waitFor(() => {
      expect(
        screen.getByText(
          'El índice de confianza del análisis de las imágenes no es suficiente para estimar la comunidad autónoma de residencia.',
        ),
      ).toBeInTheDocument();
    });
  });

  test('no avisa de confianza insuficiente cuando las fotos sí dieron una ubicación de residencia fiable', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([
      {
        done: true,
        report: makeExposureReport({
          population_narrowing: [
            {
              attribute_label: 'Vive en comunidad autónoma: Canarias',
              category: 'ubicacion',
              remaining_population: 2200000,
              risk_level: 'medio',
              evidence: ['https://instagram.com/p/1', 'https://instagram.com/p/2'],
              source: 'imagen',
              note: null,
              proportion: 0.045,
              reduction_percent: null,
            },
          ],
        }),
      },
    ]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/Vive en comunidad autónoma: Canarias/)).toBeInTheDocument();
    });
    expect(
      screen.queryByText(/El índice de confianza del análisis de las imágenes no es suficiente/),
    ).not.toBeInTheDocument();
  });

  test('no avisa de confianza insuficiente cuando no se ha analizado ninguna foto', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([{ done: true, report: makeExposureReport({ image_location_points: [] }) }]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Ubicaciones estimadas a partir de tus fotos')).toBeInTheDocument();
    });
    expect(
      screen.queryByText(/El índice de confianza del análisis de las imágenes no es suficiente/),
    ).not.toBeInTheDocument();
  });

  test('muestra el número de personas que comparten los rasgos combinados', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([{ done: true, report: makeExposureReport({ remaining_population_all_traits: 1234567 }) }]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/En España hay/)).toBeInTheDocument();
      expect(screen.getByText('1.234.567')).toBeInTheDocument();
      expect(screen.getByText(/personas que comparten tus rasgos/)).toBeInTheDocument();
    });
  });

  test('no muestra el resumen de rasgos combinados cuando no hay ningún rasgo estimable', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: true });
    mockStream([{ done: true, report: makeExposureReport({ remaining_population_all_traits: null }) }]);
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Qué se puede inferir sobre ti')).toBeInTheDocument();
    });
    expect(screen.queryByText(/personas que comparten tus rasgos/)).not.toBeInTheDocument();
  });
});
