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

  test('el botón de conexión apunta a la plataforma activa (Reddit por defecto)', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: false });

    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Landing />
      </MemoryRouter>,
    );

    const cta = await screen.findByText(/Conectar con Reddit/i);
    expect(cta.closest('a')).toHaveAttribute('href', 'http://localhost:3000/auth/reddit/login');
  });

  test('la flecha "siguiente" cambia la plataforma activa a Instagram', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: false });

    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Landing />
      </MemoryRouter>,
    );

    await screen.findByText(/Conectar con Reddit/i);
    fireEvent.click(screen.getByRole('button', { name: /Siguiente plataforma/i }));

    const cta = await screen.findByText(/Conectar con Instagram/i);
    expect(cta.closest('a')).toHaveAttribute('href', 'http://localhost:3000/auth/instagram/login');
  });

  test('la carta de X muestra "Coming Soon" y el botón queda deshabilitado', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: false });

    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Landing />
      </MemoryRouter>,
    );

    await screen.findByText(/Conectar con Reddit/i);
    const nextButton = screen.getByRole('button', { name: /Siguiente plataforma/i });
    fireEvent.click(nextButton); // -> Instagram
    fireEvent.click(nextButton); // -> X

    expect(await screen.findByText('Coming Soon')).toBeInTheDocument();

    const disabledCta = screen.getByRole('button', { name: /Próximamente/i });
    expect(disabledCta).toBeDisabled();
  });
});
