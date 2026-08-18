import '@testing-library/jest-dom';
import { render, screen, within } from '@testing-library/react';
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
    representative: true,
    created_utc: '2024-06-15T10:00:00Z',
    visual_description: 'DESCRIPCION: una persona tocando la guitarra\nPERSONAS: una\nAFICION: guitarra\nPAREJA: no',
    visual_description_general: 'una persona tocando la guitarra',
    ...overrides,
  };
}

describe('LocationMap', () => {
  test('plataforma reddit: mensaje específico de "no hay fotos", ni mapa ni lista', () => {
    render(<LocationMap points={[]} platform="reddit" available={false} />);

    expect(screen.getByText(/Reddit no tiene fotos/)).toBeInTheDocument();
    expect(screen.queryByTestId('map-container')).not.toBeInTheDocument();
    expect(screen.queryByRole('list')).not.toBeInTheDocument();
  });

  test('índice no disponible: mensaje específico, sin mencionar confianza', () => {
    render(<LocationMap points={[]} platform="instagram" available={false} />);

    expect(screen.getByText(/índice de geolocalización por imagen no está construido/)).toBeInTheDocument();
    expect(screen.queryByTestId('map-container')).not.toBeInTheDocument();
  });

  test('índice no disponible con puntos (no debería pasar, pero por si acaso): igualmente prioriza ese mensaje', () => {
    render(<LocationMap points={[makePoint()]} platform="instagram" available={false} />);

    expect(screen.getByText(/índice de geolocalización por imagen no está construido/)).toBeInTheDocument();
  });

  test('disponible pero sin fotos analizadas: mensaje distinto al de índice no disponible', () => {
    render(<LocationMap points={[]} platform="instagram" available={true} />);

    expect(screen.getByText(/No se ha podido analizar ninguna de tus fotos/)).toBeInTheDocument();
    expect(screen.queryByText(/índice de geolocalización por imagen no está construido/)).not.toBeInTheDocument();
  });

  test('puntos con lat/lon null no se pintan en el mapa, pero sí aparecen en la lista', () => {
    const points = [makePoint({ lat: null, lon: null })];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.queryByTestId('map-container')).not.toBeInTheDocument();
    expect(screen.getByText(/sin coordenadas para el mapa/)).toBeInTheDocument();
  });

  test('con puntos válidos: renderiza el mapa y un marcador por punto', () => {
    const points = [
      makePoint({ province: 'Madrid', permalink: 'https://instagram.com/p/1' }),
      makePoint({ province: 'Barcelona', lat: 41.4, lon: 2.2, permalink: 'https://instagram.com/p/2' }),
    ];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.getByTestId('map-container')).toBeInTheDocument();
    expect(screen.getAllByTestId('circle-marker')).toHaveLength(2);
  });

  test('todas las fotos, también las de baja confianza, aparecen en la lista', () => {
    const points = [
      makePoint({ permalink: 'https://instagram.com/p/alta', confidence: 0.9 }),
      makePoint({ permalink: 'https://instagram.com/p/baja', confidence: 0.1 }),
    ];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    const list = screen.getByRole('list');
    expect(within(list).getAllByRole('link', { name: 'Ver publicación' })).toHaveLength(2);
    expect(screen.getByText(/confianza 90%/)).toBeInTheDocument();
    expect(screen.getByText(/confianza 10%/)).toBeInTheDocument();
  });

  test('mapa solo incluye los puntos con coordenadas, aunque la lista los tenga todos', () => {
    const points = [
      makePoint({ lat: null, lon: null, permalink: 'https://instagram.com/p/invalido' }),
      makePoint({ province: 'Madrid', permalink: 'https://instagram.com/p/valido' }),
    ];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.getAllByTestId('circle-marker')).toHaveLength(1);
    expect(within(screen.getByRole('list')).getAllByRole('link', { name: 'Ver publicación' })).toHaveLength(2);
  });

  test('el centro del mapa es la media de lat/lon de los puntos con coordenadas', () => {
    const points = [
      makePoint({ lat: 40.0, lon: -4.0, permalink: 'https://instagram.com/p/1' }),
      makePoint({ lat: 42.0, lon: -2.0, permalink: 'https://instagram.com/p/2' }),
    ];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    const container = screen.getByTestId('map-container');
    const center = JSON.parse(container.getAttribute('data-center')!);
    expect(center).toEqual([41.0, -3.0]);
  });

  test('confianza alta (>=0.7) usa el color de riesgo más intenso', () => {
    render(<LocationMap points={[makePoint({ confidence: 0.9 })]} platform="instagram" available={true} />);
    expect(screen.getByTestId('circle-marker')).toHaveAttribute('data-color', '#d3403a');
  });

  test('confianza media (0.4-0.69) usa el color ámbar', () => {
    render(<LocationMap points={[makePoint({ confidence: 0.5 })]} platform="instagram" available={true} />);
    expect(screen.getByTestId('circle-marker')).toHaveAttribute('data-color', '#d6a51c');
  });

  test('confianza baja (<0.4) usa el color verde', () => {
    render(<LocationMap points={[makePoint({ confidence: 0.2 })]} platform="instagram" available={true} />);
    expect(screen.getByTestId('circle-marker')).toHaveAttribute('data-color', '#3aa657');
  });

  test('el radio del marcador crece con la confianza', () => {
    render(<LocationMap points={[makePoint({ confidence: 1.0 })]} platform="instagram" available={true} />);
    expect(screen.getByTestId('circle-marker')).toHaveAttribute('data-radius', '18'); // 8 + 1.0*10
  });

  test('el tooltip muestra provincia, porcentaje de confianza y coordenadas', () => {
    render(
      <LocationMap
        points={[makePoint({ province: 'Sevilla', confidence: 0.42, lat: 37.3886, lon: -5.9823 })]}
        platform="instagram"
        available={true}
      />
    );

    const tooltip = screen.getAllByTestId('tooltip')[0];
    expect(tooltip.textContent).toContain('Sevilla');
    expect(tooltip.textContent).toContain('42%');
    expect(tooltip.textContent).toContain('37.3886');
    expect(tooltip.textContent).toContain('-5.9823');
  });

  test('el popup incluye un enlace a la publicación original', () => {
    render(
      <LocationMap
        points={[makePoint({ permalink: 'https://instagram.com/p/xyz' })]}
        platform="instagram"
        available={true}
      />
    );

    const links = screen.getAllByRole('link', { name: 'Ver publicación' });
    expect(links[0]).toHaveAttribute('href', 'https://instagram.com/p/xyz');
    expect(links[0]).toHaveAttribute('target', '_blank');
  });

  test('con puntos válidos: muestra la nota explicativa sobre similitud aproximada', () => {
    render(<LocationMap points={[makePoint()]} platform="instagram" available={true} />);
    expect(screen.getByText(/similitud visual aproximada/i)).toBeInTheDocument();
  });

  test('foto no representativa: no aparece en el mapa ni en la lista principal', () => {
    const points = [makePoint({ representative: false, permalink: 'https://instagram.com/p/no-rep' })];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.queryByTestId('map-container')).not.toBeInTheDocument();
    expect(screen.queryByTestId('circle-marker')).not.toBeInTheDocument();
  });

  test('foto no representativa: aparece en el apartado "Imágenes No Representativas" con enlace a la publicación', () => {
    const points = [makePoint({ representative: false, permalink: 'https://instagram.com/p/no-rep' })];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.getByText(/Imágenes no geolocalizables/i)).toBeInTheDocument();
    const link = screen.getByRole('link', { name: 'Ver publicación' });
    expect(link).toHaveAttribute('href', 'https://instagram.com/p/no-rep');
  });

  test('mezcla de representativas y no representativas: cada una en su sitio, sin duplicarse', () => {
    const points = [
      makePoint({ permalink: 'https://instagram.com/p/rep', representative: true }),
      makePoint({ permalink: 'https://instagram.com/p/no-rep', representative: false }),
    ];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.getAllByTestId('circle-marker')).toHaveLength(1);
    const [mainList, nonRepresentativeList] = screen.getAllByRole('list');
    expect(within(mainList).getAllByRole('link', { name: 'Ver publicación' })).toHaveLength(1);
    expect(within(mainList).getByRole('link')).toHaveAttribute('href', 'https://instagram.com/p/rep');
    expect(within(nonRepresentativeList).getAllByRole('link', { name: 'Ver publicación' })).toHaveLength(1);
    expect(within(nonRepresentativeList).getByRole('link')).toHaveAttribute(
      'href',
      'https://instagram.com/p/no-rep',
    );
  });

  test('sin apartado de no representativas cuando todas las fotos son representativas', () => {
    render(<LocationMap points={[makePoint()]} platform="instagram" available={true} />);
    expect(screen.queryByText(/Imágenes no geolocalizables/i)).not.toBeInTheDocument();
  });

  test('todas las fotos analizadas son no representativas: no se muestra mapa ni lista principal, solo el aviso y el apartado aparte', () => {
    const points = [makePoint({ representative: false })];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.queryByTestId('map-container')).not.toBeInTheDocument();
    expect(
      screen.getByText(/Ninguna de tus fotos analizadas fue lo bastante representativa/),
    ).toBeInTheDocument();
    expect(screen.getByText(/Imágenes no geolocalizables/i)).toBeInTheDocument();
  });

  test('el apartado de no representativas muestra el número de fotos en el título del desplegable', () => {
    const points = [
      makePoint({ representative: false, permalink: 'https://instagram.com/p/1' }),
      makePoint({ representative: false, permalink: 'https://instagram.com/p/2' }),
      makePoint({ representative: false, permalink: 'https://instagram.com/p/3' }),
    ];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.getByText('Imágenes no geolocalizables (3)')).toBeInTheDocument();
  });

  test('el apartado de no representativas es un <details> desplegable (colapsado por defecto)', () => {
    const points = [makePoint({ representative: false })];
    const { container } = render(<LocationMap points={points} platform="instagram" available={true} />);

    const details = container.querySelector('details.image-location-details');
    expect(details).toBeInTheDocument();
    expect(details).not.toHaveAttribute('open');
  });

  test('cada foto (representativa o no) muestra su fecha de publicación', () => {
    const points = [
      makePoint({ permalink: 'https://instagram.com/p/rep', created_utc: '2024-06-15T10:00:00Z' }),
      makePoint({
        permalink: 'https://instagram.com/p/no-rep',
        representative: false,
        created_utc: '2023-01-02T10:00:00Z',
      }),
    ];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.getByText(/15 jun 2024/)).toBeInTheDocument();
    expect(screen.getByText(/2 ene 2023/)).toBeInTheDocument();
  });

  test('fecha desconocida cuando created_utc es null, sin romper el resto de la fila', () => {
    const points = [makePoint({ created_utc: null })];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.getByText(/fecha desconocida/)).toBeInTheDocument();
  });

  test('cada foto (representativa o no) es un <details> propio, colapsado por defecto', () => {
    const points = [
      makePoint({ permalink: 'https://instagram.com/p/rep' }),
      makePoint({ permalink: 'https://instagram.com/p/no-rep', representative: false }),
    ];
    const { container } = render(<LocationMap points={points} platform="instagram" available={true} />);

    const photoDetails = container.querySelectorAll('details.photo-details');
    expect(photoDetails).toHaveLength(2);
    photoDetails.forEach((el) => expect(el).not.toHaveAttribute('open'));
  });

  test('al desplegar una foto se ve la descripción de Moondream2', () => {
    const points = [makePoint({ visual_description: 'PERSONAS: una\nAFICION: baloncesto\nPAREJA: no' })];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.getByText(/AFICION: baloncesto/)).toBeInTheDocument();
  });

  test('sin descripción visual (null): muestra el aviso en vez de dejarlo vacío', () => {
    const points = [makePoint({ visual_description: null })];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.getByText(/Sin descripción visual disponible/)).toBeInTheDocument();
  });

  test('al desplegar una foto se ve la descripción general', () => {
    const points = [makePoint({ visual_description_general: '4 personas comiendo pizza alegremente en una terraza' })];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.getByText('4 personas comiendo pizza alegremente en una terraza')).toBeInTheDocument();
  });

  test('sin descripción general (null): muestra el aviso en vez de dejarlo vacío', () => {
    const points = [makePoint({ visual_description_general: null })];
    render(<LocationMap points={points} platform="instagram" available={true} />);

    expect(screen.getByText(/Sin descripción general disponible/)).toBeInTheDocument();
  });

  test('el apartado de no representativas tiene la clase de scroll con altura máxima', () => {
    const points = [makePoint({ representative: false })];
    const { container } = render(<LocationMap points={points} platform="instagram" available={true} />);

    const scrollList = container.querySelector('.image-location-list.image-location-list-scroll');
    expect(scrollList).toBeInTheDocument();
  });

  describe('foto de perfil (is_profile_picture)', () => {
    test('con permalink: se muestra como enlace con la etiqueta "Foto de perfil", no "Ver publicación"', () => {
      const points = [
        makePoint({ is_profile_picture: true, permalink: 'https://cdn.fake/avatar.jpg', created_utc: null }),
      ];
      render(<LocationMap points={points} platform="instagram" available={true} />);

      const links = screen.getAllByRole('link', { name: 'Foto de perfil' });
      expect(links.length).toBeGreaterThan(0);
      links.forEach((link) => expect(link).toHaveAttribute('href', 'https://cdn.fake/avatar.jpg'));
      expect(screen.queryByText('Ver publicación')).not.toBeInTheDocument();
    });

    test('sin permalink: texto suelto "Foto de perfil", sin enlace', () => {
      // Sin lat/lon: no entra en el mapa (que también renderizaría un
      // enlace en el popup), así que solo aparece en la lista principal.
      const points = [
        makePoint({ is_profile_picture: true, permalink: '', lat: null, lon: null, created_utc: null }),
      ];
      const { container } = render(<LocationMap points={points} platform="instagram" available={true} />);

      expect(screen.getByText('Foto de perfil')).toBeInTheDocument();
      expect(container.querySelector('.photo-details summary a')).not.toBeInTheDocument();
      expect(container.querySelector('.photo-link-label')).toBeInTheDocument();
    });

    test('foto normal (is_profile_picture ausente/false): sigue diciendo "Ver publicación"', () => {
      const points = [makePoint()];
      render(<LocationMap points={points} platform="instagram" available={true} />);

      expect(screen.getAllByRole('link', { name: 'Ver publicación' }).length).toBeGreaterThan(0);
      expect(screen.queryByText('Foto de perfil')).not.toBeInTheDocument();
    });

    test('en el mapa (popup), la foto de perfil con link también dice "Foto de perfil"', () => {
      const points = [makePoint({ is_profile_picture: true, permalink: 'https://cdn.fake/avatar.jpg' })];
      render(<LocationMap points={points} platform="instagram" available={true} />);

      const popup = within(screen.getByTestId('popup'));
      expect(popup.getByRole('link', { name: 'Foto de perfil' })).toHaveAttribute(
        'href',
        'https://cdn.fake/avatar.jpg',
      );
    });

    test('foto de perfil no representativa: también se etiqueta correctamente en esa lista', () => {
      const points = [
        makePoint({
          is_profile_picture: true,
          representative: false,
          permalink: 'https://cdn.fake/avatar.jpg',
        }),
      ];
      render(<LocationMap points={points} platform="instagram" available={true} />);

      expect(screen.getByRole('link', { name: 'Foto de perfil' })).toBeInTheDocument();
    });
  });
});
