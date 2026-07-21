import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { describe, expect, test, vi } from 'vitest';
import LocationMap from '../components/LocationMap';
import type { ImageLocationPoint } from '../types';

// react-leaflet depende de APIs de navegador real (mediciones de DOM,
// tiles) que jsdom no implementa de forma fiable. Se sustituye por stubs
// ligeros que exponen las props relevantes como texto/atributos
// testeables, para verificar LA LÓGICA de LocationMap (filtrado de puntos,
// cálculo de centro, umbrales de color por confianza, estado vacío) sin
// depender de las tripas de Leaflet, que no es código propio del proyecto.
vi.mock('react-leaflet', () => ({
  MapContainer: ({ center, children }: any) => (
    <div data-testid="map-container" data-center={JSON.stringify(center)}>
      {children}
    </div>
  ),
  TileLayer: () => <div data-testid="tile-layer" />,
  CircleMarker: ({ center, radius, pathOptions, children }: any) => (
    <div
      data-testid="circle-marker"
      data-center={JSON.stringify(center)}
      data-radius={radius}
      data-color={pathOptions?.color}
    >
      {children}
    </div>
  ),
  Tooltip: ({ children }: any) => <div data-testid="tooltip">{children}</div>,
  Popup: ({ children }: any) => <div data-testid="popup">{children}</div>,
}));

function makePoint(overrides: Partial<ImageLocationPoint> = {}): ImageLocationPoint {
  return {
    permalink: 'https://instagram.com/p/1',
    province: 'Madrid',
    confidence: 0.6,
    lat: 40.41,
    lon: -3.7,
    ...overrides,
  };
}

describe('LocationMap', () => {
  test('sin puntos válidos: muestra el mensaje explicativo, no el mapa', () => {
    render(<LocationMap points={[]} />);

    expect(screen.getByText(/No se ha podido estimar la ubicación/)).toBeInTheDocument();
    expect(screen.queryByTestId('map-container')).not.toBeInTheDocument();
  });

  test('puntos con lat/lon null se filtran y no cuentan como válidos', () => {
    const points = [makePoint({ lat: null, lon: null })];
    render(<LocationMap points={points} />);

    expect(screen.getByText(/No se ha podido estimar la ubicación/)).toBeInTheDocument();
  });

  test('con puntos válidos: renderiza el mapa y un marcador por punto', () => {
    const points = [
      makePoint({ province: 'Madrid', permalink: 'https://instagram.com/p/1' }),
      makePoint({ province: 'Barcelona', lat: 41.4, lon: 2.2, permalink: 'https://instagram.com/p/2' }),
    ];
    render(<LocationMap points={points} />);

    expect(screen.getByTestId('map-container')).toBeInTheDocument();
    expect(screen.getAllByTestId('circle-marker')).toHaveLength(2);
  });

  test('filtra puntos inválidos pero conserva y renderiza los válidos', () => {
    const points = [
      makePoint({ lat: null, lon: null, permalink: 'https://instagram.com/p/invalido' }),
      makePoint({ province: 'Madrid', permalink: 'https://instagram.com/p/valido' }),
    ];
    render(<LocationMap points={points} />);

    expect(screen.getAllByTestId('circle-marker')).toHaveLength(1);
  });

  test('el centro del mapa es la media de lat/lon de los puntos válidos', () => {
    const points = [
      makePoint({ lat: 40.0, lon: -4.0, permalink: 'https://instagram.com/p/1' }),
      makePoint({ lat: 42.0, lon: -2.0, permalink: 'https://instagram.com/p/2' }),
    ];
    render(<LocationMap points={points} />);

    const container = screen.getByTestId('map-container');
    const center = JSON.parse(container.getAttribute('data-center')!);
    expect(center).toEqual([41.0, -3.0]);
  });

  test('confianza alta (>=0.7) usa el color de riesgo más intenso', () => {
    render(<LocationMap points={[makePoint({ confidence: 0.9 })]} />);
    expect(screen.getByTestId('circle-marker')).toHaveAttribute('data-color', '#d3403a');
  });

  test('confianza media (0.4-0.69) usa el color ámbar', () => {
    render(<LocationMap points={[makePoint({ confidence: 0.5 })]} />);
    expect(screen.getByTestId('circle-marker')).toHaveAttribute('data-color', '#d6a51c');
  });

  test('confianza baja (<0.4) usa el color verde', () => {
    render(<LocationMap points={[makePoint({ confidence: 0.2 })]} />);
    expect(screen.getByTestId('circle-marker')).toHaveAttribute('data-color', '#3aa657');
  });

  test('el radio del marcador crece con la confianza', () => {
    render(<LocationMap points={[makePoint({ confidence: 1.0 })]} />);
    expect(screen.getByTestId('circle-marker')).toHaveAttribute('data-radius', '18'); // 8 + 1.0*10
  });

  test('el tooltip muestra provincia, porcentaje de confianza y coordenadas', () => {
    render(
      <LocationMap points={[makePoint({ province: 'Sevilla', confidence: 0.42, lat: 37.3886, lon: -5.9823 })]} />
    );

    const tooltip = screen.getAllByTestId('tooltip')[0];
    expect(tooltip.textContent).toContain('Sevilla');
    expect(tooltip.textContent).toContain('42%');
    expect(tooltip.textContent).toContain('37.3886');
    expect(tooltip.textContent).toContain('-5.9823');
  });

  test('el popup incluye un enlace a la publicación original', () => {
    render(<LocationMap points={[makePoint({ permalink: 'https://instagram.com/p/xyz' })]} />);

    const link = screen.getByRole('link', { name: 'Ver publicación' });
    expect(link).toHaveAttribute('href', 'https://instagram.com/p/xyz');
    expect(link).toHaveAttribute('target', '_blank');
  });

  test('con puntos válidos: muestra la nota explicativa sobre precisión aproximada', () => {
    render(<LocationMap points={[makePoint()]} />);
    expect(screen.getByText(/estimación aproximada/i)).toBeInTheDocument();
  });
});
