import '@testing-library/jest-dom';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, test, vi } from 'vitest';
import { api } from '../api';
import Landing from '../pages/Landing';

vi.mock('../api', () => ({
  api: {
    authStatus: vi.fn(),
    loginUrl: () => 'http://localhost:3000/auth/reddit/login',
  },
}));

describe('Landing', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('muestra el aviso de consentimiento antes de conectar con Reddit', async () => {
    vi.mocked(api.authStatus).mockResolvedValueOnce({ authenticated: false });

    render(
      <MemoryRouter>
        <Landing />
      </MemoryRouter>,
    );

    expect(
      screen.getByText(/Solo se analiza tu propia cuenta de Reddit/i),
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/Conectar con Reddit y empezar/i)).toBeInTheDocument();
    });
  });
});
