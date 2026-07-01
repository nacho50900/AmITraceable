import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { describe, expect, test } from 'vitest';
import ScoreBar from '../components/ScoreBar';

describe('ScoreBar', () => {
  test('etiqueta el riesgo como "Bajo" cuando el valor es reducido', () => {
    render(<ScoreBar label="Riesgo de prueba" value={10} />);
    expect(screen.getByText('Riesgo de prueba')).toBeInTheDocument();
    expect(screen.getByText(/Bajo \(10\.0\)/)).toBeInTheDocument();
  });

  test('etiqueta el riesgo como "Muy alto" cuando el valor es cercano a 100', () => {
    render(<ScoreBar label="Riesgo de prueba" value={95} />);
    expect(screen.getByText(/Muy alto \(95\.0\)/)).toBeInTheDocument();
  });

  test('limita visualmente el ancho de la barra al 100% aunque el valor lo supere', () => {
    render(<ScoreBar label="Riesgo de prueba" value={140} />);
    const track = screen.getByText('Riesgo de prueba').closest('.score-row');
    const fill = track?.querySelector('.score-fill') as HTMLElement;
    expect(fill.style.width).toBe('100%');
  });
});
