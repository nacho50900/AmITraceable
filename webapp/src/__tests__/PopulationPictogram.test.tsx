import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { describe, expect, test } from 'vitest';
import PopulationPictogram from '../components/PopulationPictogram';

describe('PopulationPictogram', () => {
  test('proportion o remainingPopulation nulos: no renderiza nada', () => {
    const { container: c1 } = render(<PopulationPictogram proportion={null} remainingPopulation={1000} />);
    expect(c1).toBeEmptyDOMElement();

    const { container: c2 } = render(<PopulationPictogram proportion={0.5} remainingPopulation={null} />);
    expect(c2).toBeEmptyDOMElement();
  });

  test('50%: se muestra la etiqueta de porcentaje', () => {
    render(<PopulationPictogram proportion={0.5} remainingPopulation={24_500_000} />);
    expect(screen.getByText('50%')).toBeInTheDocument();
  });

  test('50%: exactamente 2 imágenes, 1 turquesa y 1 negra', () => {
    const { container } = render(<PopulationPictogram proportion={0.5} remainingPopulation={24_500_000} />);
    const images = container.querySelectorAll('img');

    expect(images).toHaveLength(2);
    expect(images[0]).toHaveAttribute('src', '/monigote-selected.png');
    expect(images[1]).toHaveAttribute('src', '/monigote.png');
  });

  test('10%: 1 monigote turquesa y 9 negros (10 en total)', () => {
    const { container } = render(<PopulationPictogram proportion={0.1} remainingPopulation={4_900_000} />);
    const images = container.querySelectorAll('img');

    expect(images).toHaveLength(10);
    expect(images[0]).toHaveAttribute('src', '/monigote-selected.png');
    expect(Array.from(images).slice(1).every((img) => img.getAttribute('src') === '/monigote.png')).toBe(true);
    expect(screen.getByText('10%')).toBeInTheDocument();
  });

  test('proporción muy pequeña (menos del 1%, más de 100 monigotes): modo compacto con el número absoluto', () => {
    const { container } = render(<PopulationPictogram proportion={0.0035} remainingPopulation={170000} />);
    const images = container.querySelectorAll('img');

    expect(images).toHaveLength(1);
    expect(images[0]).toHaveAttribute('src', '/monigote-selected.png');
    expect(screen.getByText('de 170.000')).toBeInTheDocument();
  });

  test('proporción en el límite (exactamente 100 monigotes): sigue en modo rejilla', () => {
    const { container } = render(<PopulationPictogram proportion={0.01} remainingPopulation={491282} />);
    const images = container.querySelectorAll('img');

    expect(images).toHaveLength(100);
  });

  test('etiqueta de accesibilidad describe la proporción en texto', () => {
    render(<PopulationPictogram proportion={0.25} remainingPopulation={12_282_074} />);
    expect(screen.getByRole('img', { name: /25% de la población/ })).toBeInTheDocument();
  });

  test('formatea el número absoluto con separador de miles en español (modo compacto)', () => {
    render(<PopulationPictogram proportion={0.001} remainingPopulation={49128} />);
    expect(screen.getByText('de 49.128')).toBeInTheDocument();
  });
});
