import '@testing-library/jest-dom';
import { render, screen, waitFor } from '@testing-library/react';
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

  test('muestra el aviso de consentimiento y ambas opciones de plataforma', async () => {
    vi.mocked(api.authStatus).mockResolvedValue({ authenticated: false });

    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Landing />
      </MemoryRouter>,
    );

    expect(screen.getByText('AmITraceable')).toBeInTheDocument();
    expect(screen.getByText(/Solo se analiza tu propia cuenta/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/Conectar con Reddit/i)).toBeInTheDocument();
      expect(screen.getByText(/Conectar con Instagram/i)).toBeInTheDocument();
    });
  });
});
