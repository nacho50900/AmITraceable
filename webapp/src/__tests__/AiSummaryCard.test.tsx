import '@testing-library/jest-dom';
import { act, render, screen, waitFor } from '@testing-library/react';
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

    expect(screen.getByText('Recomendaciones y Conclusiones')).toBeInTheDocument();
    expect(screen.getByText('Analizando el informe con IA...')).toBeInTheDocument();
    expect(api.aiSummary).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('button', { name: 'Analizar con IA' })).not.toBeInTheDocument();
  });

  test('un informe NUEVO objeto pero del mismo análisis (p. ej. tras traducir descripciones) NO relanza la IA', async () => {
    vi.mocked(api.aiSummary).mockImplementation(() => new Promise(() => {}));
    const report = makeExposureReport();
    const { rerender } = render(<AiSummaryCard report={report} />);

    // Mismo análisis (mismo generated_at), objeto distinto: lo que hace el
    // Dashboard al terminar la traducción de las descripciones.
    rerender(<AiSummaryCard report={{ ...report }} />);
    rerender(<AiSummaryCard report={{ ...report, image_location_points: [] }} />);

    expect(api.aiSummary).toHaveBeenCalledTimes(1);
  });

  test('un informe de OTRO análisis (generated_at distinto, p. ej. recálculo) sí relanza la IA', async () => {
    vi.mocked(api.aiSummary).mockImplementation(() => new Promise(() => {}));
    const report = makeExposureReport();
    const { rerender } = render(<AiSummaryCard report={report} />);

    rerender(<AiSummaryCard report={{ ...report, generated_at: '2099-01-01T00:00:00Z' }} />);

    expect(api.aiSummary).toHaveBeenCalledTimes(2);
  });

  test('una respuesta obsoleta (de una petición anterior) no pisa a la más reciente', async () => {
    let resolveFirst: (v: { verdict: string; conclusions: string[] }) => void = () => {};
    vi.mocked(api.aiSummary)
      .mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve; }))
      .mockResolvedValueOnce({ verdict: 'Veredicto NUEVO', conclusions: [] });
    const report = makeExposureReport();
    const { rerender } = render(<AiSummaryCard report={report} />);

    rerender(<AiSummaryCard report={{ ...report, generated_at: '2099-01-01T00:00:00Z' }} />);
    await waitFor(() => expect(screen.getByText('Veredicto NUEVO')).toBeInTheDocument());

    await act(async () => {
      resolveFirst({ verdict: 'Veredicto VIEJO', conclusions: [] });
    });

    expect(screen.getByText('Veredicto NUEVO')).toBeInTheDocument();
    expect(screen.queryByText('Veredicto VIEJO')).not.toBeInTheDocument();
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

  test('no disponible (503): muestra el mensaje y permite reintentar', async () => {
    vi.mocked(api.aiSummary).mockRejectedValue(new AiSummaryUnavailableError('Cuota agotada por hoy.'));
    render(<AiSummaryCard report={makeExposureReport()} />);

    await waitFor(() => {
      expect(screen.getByText(/Cuota agotada por hoy\./)).toBeInTheDocument();
    });
    // Antes de este fix no había botón aquí: un 503 por el modelo ocupado
    // (ver lock_timeout en app/nlp/ai_client.py) dejaba al usuario sin
    // forma de reintentar sin recargar la página entera.
    expect(screen.getByRole('button', { name: 'Reintentar' })).toBeInTheDocument();
  });

  test('reintentar tras "no disponible" vuelve a llamar a la API y puede tener éxito', async () => {
    const user = userEvent.setup();
    vi.mocked(api.aiSummary)
      .mockRejectedValueOnce(new AiSummaryUnavailableError('El modelo está ocupado.'))
      .mockResolvedValueOnce({ verdict: '', conclusions: ['Conclusión tras reintento'] });
    render(<AiSummaryCard report={makeExposureReport()} />);

    const retryButton = await screen.findByRole('button', { name: 'Reintentar' });
    await user.click(retryButton);

    await waitFor(() => {
      expect(screen.getByText('Conclusión tras reintento')).toBeInTheDocument();
    });
    expect(api.aiSummary).toHaveBeenCalledTimes(2);
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
