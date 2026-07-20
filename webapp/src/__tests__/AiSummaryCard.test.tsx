import '@testing-library/jest-dom';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, test, vi } from 'vitest';
import { AiSummaryUnavailableError, api } from '../api';
import AiSummaryCard from '../components/AiSummaryCard';
import { makeExposureReport } from './fixtures';

vi.mock('../api', async () => {
  const actual = await vi.importActual<typeof import('../api')>('../api');
  return {
    ...actual,
    api: { ...actual.api, aiSummary: vi.fn() },
  };
});

// El proyecto no tiene clearMocks/restoreMocks activado globalmente en
// vitest.config, así que sin este reset explícito las implementaciones de
// mock (mockResolvedValueOnce, etc.) se acumulan entre tests de este fichero.
beforeEach(() => {
  vi.mocked(api.aiSummary).mockReset();
});

describe('AiSummaryCard', () => {
  test('estado inicial: muestra explicación y botón "Analizar con IA"', () => {
    render(<AiSummaryCard report={makeExposureReport()} />);

    expect(screen.getByText('Conclusiones generadas por IA')).toBeInTheDocument();
    expect(screen.getByText(/Mistral AI/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analizar con IA' })).toBeInTheDocument();
  });

  test('al pulsar el botón, muestra el estado de carga', async () => {
    const user = userEvent.setup();
    vi.mocked(api.aiSummary).mockImplementation(() => new Promise(() => {})); // nunca resuelve
    render(<AiSummaryCard report={makeExposureReport()} />);

    await user.click(screen.getByRole('button', { name: 'Analizar con IA' }));

    expect(screen.getByText('Generando conclusiones...')).toBeInTheDocument();
  });

  test('éxito: muestra la lista de conclusiones devueltas', async () => {
    const user = userEvent.setup();
    vi.mocked(api.aiSummary).mockResolvedValue({ conclusions: ['Primera conclusión', 'Segunda conclusión'] });
    render(<AiSummaryCard report={makeExposureReport()} />);

    await user.click(screen.getByRole('button', { name: 'Analizar con IA' }));

    await waitFor(() => {
      expect(screen.getByText('Primera conclusión')).toBeInTheDocument();
      expect(screen.getByText('Segunda conclusión')).toBeInTheDocument();
    });
  });

  test('éxito: llama a api.aiSummary con el informe exacto recibido por props', async () => {
    const user = userEvent.setup();
    const report = makeExposureReport({ username: 'otro_usuario' });
    vi.mocked(api.aiSummary).mockResolvedValue({ conclusions: ['x'] });
    render(<AiSummaryCard report={report} />);

    await user.click(screen.getByRole('button', { name: 'Analizar con IA' }));

    await waitFor(() => expect(api.aiSummary).toHaveBeenCalledWith(report));
  });

  test('no disponible (503): muestra el mensaje del error sin botón de reintento', async () => {
    const user = userEvent.setup();
    vi.mocked(api.aiSummary).mockRejectedValue(new AiSummaryUnavailableError('Cuota agotada por hoy.'));
    render(<AiSummaryCard report={makeExposureReport()} />);

    await user.click(screen.getByRole('button', { name: 'Analizar con IA' }));

    await waitFor(() => {
      expect(screen.getByText(/Cuota agotada por hoy\./)).toBeInTheDocument();
      expect(screen.getByText(/función complementaria opcional/)).toBeInTheDocument();
    });
    expect(screen.queryByRole('button', { name: 'Reintentar' })).not.toBeInTheDocument();
  });

  test('error genérico: muestra mensaje de error con botón de reintento', async () => {
    const user = userEvent.setup();
    vi.mocked(api.aiSummary).mockRejectedValue(new Error('fallo de red'));
    render(<AiSummaryCard report={makeExposureReport()} />);

    await user.click(screen.getByRole('button', { name: 'Analizar con IA' }));

    await waitFor(() => {
      expect(screen.getByText(/fallo de red/)).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: 'Reintentar' })).toBeInTheDocument();
  });

  test('error no-Error (valor no estándar lanzado): usa mensaje genérico de fallback', async () => {
    const user = userEvent.setup();
    vi.mocked(api.aiSummary).mockRejectedValue('algo raro, no es un Error');
    render(<AiSummaryCard report={makeExposureReport()} />);

    await user.click(screen.getByRole('button', { name: 'Analizar con IA' }));

    await waitFor(() => {
      expect(screen.getByText(/Error inesperado\./)).toBeInTheDocument();
    });
  });

  test('reintentar tras un error vuelve a llamar a la API y puede tener éxito', async () => {
    const user = userEvent.setup();
    vi.mocked(api.aiSummary)
      .mockRejectedValueOnce(new Error('fallo de red'))
      .mockResolvedValueOnce({ conclusions: ['Conclusión tras reintento'] });
    render(<AiSummaryCard report={makeExposureReport()} />);

    await user.click(screen.getByRole('button', { name: 'Analizar con IA' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Reintentar' })).toBeInTheDocument());

    await user.click(screen.getByRole('button', { name: 'Reintentar' }));

    await waitFor(() => {
      expect(screen.getByText('Conclusión tras reintento')).toBeInTheDocument();
    });
    expect(api.aiSummary).toHaveBeenCalledTimes(2);
  });
});
