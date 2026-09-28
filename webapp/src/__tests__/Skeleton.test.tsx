import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, test, vi } from 'vitest';
import { api } from '../api';
import Dashboard from '../pages/Dashboard';
import Landing from '../pages/Landing';
import { DashboardSkeleton, PlatformPickerSkeleton, Skeleton, SkeletonText } from '../components/Skeleton';

vi.mock('../api', () => ({
  api: {
    authStatus: vi.fn(),
    analyzeStream: vi.fn(() => () => {}),
    loginUrl: (platform: string) => `http://localhost:3000/auth/${platform}/login`,
  },
}));

describe('Skeleton', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('Skeleton es decorativo y aplica las medidas recibidas', () => {
    const { container } = render(<Skeleton width={120} height="2rem" radius={4} className="extra" />);
    const el = container.querySelector('.skeleton') as HTMLElement;

    expect(el).toHaveAttribute('aria-hidden', 'true');
    expect(el).toHaveClass('extra');
    expect(el.style.width).toBe('120px');
    expect(el.style.height).toBe('2rem');
    expect(el.style.borderRadius).toBe('4px');
  });

  test('SkeletonText pinta las líneas pedidas y acorta la última', () => {
    const { container } = render(<SkeletonText lines={4} lastLineWidth="40%" />);
    const lines = container.querySelectorAll<HTMLElement>('.skeleton');

    expect(lines).toHaveLength(4);
    expect(lines[3].style.width).toBe('40%');
    expect(lines[0].style.width).toBe('100%');
  });

  test('SkeletonText de una sola línea no la acorta', () => {
    const { container } = render(<SkeletonText lines={1} />);

    expect((container.querySelector('.skeleton') as HTMLElement).style.width).toBe('100%');
  });

  test('PlatformPickerSkeleton y DashboardSkeleton no exponen contenido a lectores de pantalla', () => {
    render(
      <>
        <PlatformPickerSkeleton />
        <DashboardSkeleton />
      </>,
    );

    expect(screen.getByTestId('platform-picker-skeleton')).toHaveAttribute('aria-hidden', 'true');
    expect(screen.getByTestId('dashboard-skeleton')).toHaveAttribute('aria-hidden', 'true');
  });

  test('Landing muestra el skeleton del selector mientras comprueba la sesión', () => {
    vi.mocked(api.authStatus).mockImplementation(() => new Promise(() => {})); // nunca resuelve

    const { container } = render(
      <MemoryRouter>
        <Landing />
      </MemoryRouter>,
    );

    expect(screen.getByTestId('platform-picker-skeleton')).toBeInTheDocument();
    expect(container.querySelector('.card-deck')).toBeNull();
    expect(container.querySelector('.landing')).toHaveAttribute('aria-busy', 'true');
  });

  test('Dashboard muestra el skeleton del informe mientras el análisis está en curso', () => {
    vi.mocked(api.authStatus).mockImplementation(() => new Promise(() => {})); // nunca resuelve

    const { container } = render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>,
    );

    expect(screen.getByTestId('dashboard-skeleton')).toBeInTheDocument();
    expect(container.querySelector('.page')).toHaveAttribute('aria-busy', 'true');
  });
});
