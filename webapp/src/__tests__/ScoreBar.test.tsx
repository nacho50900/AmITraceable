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

  test('sin tooltip no se muestra ningún icono de información', () => {
    render(<ScoreBar label="Riesgo de prueba" value={10} />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });

  test('con tooltip se muestra un botón de información con la descripción asociada', () => {
    render(<ScoreBar label="Riesgo de prueba" value={10} tooltip="Explicación de prueba" />);
    const button = screen.getByRole('button', { name: 'Más información sobre Riesgo de prueba' });
    expect(button).toHaveAccessibleDescription('Explicación de prueba');
    expect(button.querySelector('svg')).toHaveAttribute('aria-hidden', 'true');
  });
});
