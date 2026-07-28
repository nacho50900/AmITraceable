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
  test('se dispara solo al montar, sin necesidad de pulsar nada', () => {
    vi.mocked(api.aiSummary).mockImplementation(() => new Promise(() => {})); // nunca resuelve
    render(<AiSummaryCard report={makeExposureReport()} />);

    expect(screen.getByText('Conclusiones generadas por IA')).toBeInTheDocument();
    expect(screen.getByText('Analizando el informe con IA...')).toBeInTheDocument();
    expect(api.aiSummary).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('button', { name: 'Analizar con IA' })).not.toBeInTheDocument();
  });

  test('éxito: muestra el veredicto general y la lista de conclusiones', async () => {
    vi.mocked(api.aiSummary).mockResolvedValue({
      verdict: 'Este perfil no comparte información que permita identificarte con facilidad.',
      conclusions: ['Primera conclusión', 'Segunda conclusión'],
    });
    render(<AiSummaryCard report={makeExposureReport()} />);

    await waitFor(() => {
      expect(
        screen.getByText('Este perfil no comparte información que permita identificarte con facilidad.')
      ).toBeInTheDocument();
      expect(screen.getByText('Primera conclusión')).toBeInTheDocument();
      expect(screen.getByText('Segunda conclusión')).toBeInTheDocument();
    });
  });

  test('éxito: llama a api.aiSummary con el informe exacto recibido por props', async () => {
    const report = makeExposureReport({ username: 'otro_usuario' });
    vi.mocked(api.aiSummary).mockResolvedValue({ verdict: 'Veredicto.', conclusions: [] });
    render(<AiSummaryCard report={report} />);

    await waitFor(() => expect(api.aiSummary).toHaveBeenCalledWith(report));
  });

  test('veredicto sin conclusiones adicionales: se muestra igualmente el veredicto, sin lista', async () => {
    vi.mocked(api.aiSummary).mockResolvedValue({ verdict: 'Riesgo bajo en general.', conclusions: [] });
    const { container } = render(<AiSummaryCard report={makeExposureReport()} />);

    await waitFor(() => {
      expect(screen.getByText('Riesgo bajo en general.')).toBeInTheDocument();
    });
    expect(container.querySelector('.ai-conclusions-list')).not.toBeInTheDocument();
  });

  test('respuesta totalmente vacía (sin veredicto ni conclusiones): la IA no encontró nada que destacar', async () => {
    vi.mocked(api.aiSummary).mockResolvedValue({ verdict: '', conclusions: [] });
    render(<AiSummaryCard report={makeExposureReport()} />);

    await waitFor(() => {
      expect(screen.getByText(/no ha encontrado ninguna conclusión que merezca la pena/)).toBeInTheDocument();
    });
  });

  test('no disponible (503): muestra el mensaje del error sin botón de reintento', async () => {
    vi.mocked(api.aiSummary).mockRejectedValue(new AiSummaryUnavailableError('Cuota agotada por hoy.'));
    render(<AiSummaryCard report={makeExposureReport()} />);

    await waitFor(() => {
      expect(screen.getByText(/Cuota agotada por hoy\./)).toBeInTheDocument();
    });
    expect(screen.queryByRole('button', { name: 'Reintentar' })).not.toBeInTheDocument();
  });

  test('error genérico: muestra mensaje de error con botón de reintento', async () => {
    vi.mocked(api.aiSummary).mockRejectedValue(new Error('fallo de red'));
    render(<AiSummaryCard report={makeExposureReport()} />);

    await waitFor(() => {
      expect(screen.getByText(/fallo de red/)).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: 'Reintentar' })).toBeInTheDocument();
  });

  test('error no-Error (valor no estándar lanzado): usa mensaje genérico de fallback', async () => {
    vi.mocked(api.aiSummary).mockRejectedValue('algo raro, no es un Error');
    render(<AiSummaryCard report={makeExposureReport()} />);

    await waitFor(() => {
      expect(screen.getByText(/Error inesperado\./)).toBeInTheDocument();
    });
  });

  test('reintentar tras un error vuelve a llamar a la API y puede tener éxito', async () => {
    const user = userEvent.setup();
    vi.mocked(api.aiSummary)
      .mockRejectedValueOnce(new Error('fallo de red'))
      .mockResolvedValueOnce({ verdict: '', conclusions: ['Conclusión tras reintento'] });
    render(<AiSummaryCard report={makeExposureReport()} />);

    const retryButton = await screen.findByRole('button', { name: 'Reintentar' });
    await user.click(retryButton);

    await waitFor(() => {
      expect(screen.getByText('Conclusión tras reintento')).toBeInTheDocument();
    });
    expect(api.aiSummary).toHaveBeenCalledTimes(2);
  });
});
