import '@testing-library/jest-dom';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, test, vi } from 'vitest';
import { api } from '../api';
import Landing from '../pages/Landing';

vi.mock('../api', () => ({
  api: {
    authStatus: vi.fn(),
    loginUrl: (platform: string) => `http://localhost:3000/auth/${platform}/login`,
  },
}));

describe('Landing', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('muestra el aviso de consentimiento y las tres cartas de plataforma', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: false });

    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Landing />
      </MemoryRouter>,
    );

    expect(screen.getByText('AmITraceable')).toBeInTheDocument();
    expect(screen.getByText(/Solo se analiza tu propia cuenta/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Reddit')).toBeInTheDocument();
      expect(screen.getByText('Instagram')).toBeInTheDocument();
      expect(screen.getByText('X')).toBeInTheDocument();
    });
  });

  test('la carta de Reddit muestra "Coming Soon" y el botón queda deshabilitado (integración desactivada temporalmente)', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: false });

    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Landing />
      </MemoryRouter>,
    );

    // Reddit es la carta activa por defecto (primera del mazo). El CTA de
    // debajo del mazo corresponde SIEMPRE a la carta activa (Reddit aquí),
    // así que solo hay un botón "Próximamente" aunque el badge "Coming
    // Soon" pueda aparecer más de una vez en pantalla (con 3 cartas, la
    // adyacente -X- también es visible al mismo tiempo y también está
    // desactivada -- ver el siguiente test).
    const disabledCta = await screen.findByRole('button', { name: /Próximamente/i });
    expect(disabledCta).toBeDisabled();
    expect(screen.queryByText(/Conectar con Reddit/i)).not.toBeInTheDocument();
  });

  test('el botón de conexión apunta a la plataforma activa (Instagram, tras avanzar desde Reddit)', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: false });

    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Landing />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /Próximamente/i });
    fireEvent.click(screen.getByRole('button', { name: /Siguiente plataforma/i }));

    const cta = await screen.findByText(/Conectar con Instagram/i);
    expect(cta.closest('a')).toHaveAttribute('href', 'http://localhost:3000/auth/instagram/login');
  });

  test('la carta de X también muestra "Coming Soon" y el botón queda deshabilitado', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: false });

    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Landing />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /Próximamente/i }); // Reddit, activa por defecto
    const nextButton = screen.getByRole('button', { name: /Siguiente plataforma/i });
    fireEvent.click(nextButton); // -> Instagram
    fireEvent.click(nextButton); // -> X

    // Ahora Reddit e Instagram son las cartas adyacentes visibles; solo
    // Reddit sigue mostrando su propio badge "Coming Soon" (Instagram no
    // está desactivada) -- junto con la de X (activa), hay 2 badges en
    // pantalla, no una única.
    expect(await screen.findAllByText('Coming Soon')).toHaveLength(2);
    const disabledCta = screen.getByRole('button', { name: /Próximamente/i });
    expect(disabledCta).toBeDisabled();
    expect(screen.queryByText(/Conectar con X/i)).not.toBeInTheDocument();
  });
});
