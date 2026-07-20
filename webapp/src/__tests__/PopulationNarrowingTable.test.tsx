import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { describe, expect, test } from 'vitest';
import PopulationNarrowingTable from '../components/PopulationNarrowingTable';
import type { PopulationEstimate } from '../types';

function makeStep(overrides: Partial<PopulationEstimate> = {}): PopulationEstimate {
  return {
    attribute_label: 'Sexo: mujer',
    category: 'sexo',
    remaining_population: 24_957_175,
    risk_level: 'bajo',
    evidence: ['https://x/1'],
    source: 'texto',
    note: null,
    ...overrides,
  };
}

describe('PopulationNarrowingTable', () => {
  test('lista vacía: muestra el mensaje explicativo, no la tabla', () => {
    render(<PopulationNarrowingTable steps={[]} />);

    expect(screen.getByText(/No se han detectado declaraciones explícitas/)).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  test('renderiza una fila por cada paso, con las 4 columnas', () => {
    const steps = [makeStep({ attribute_label: 'Sexo: mujer' }), makeStep({ attribute_label: 'Edad: 24 años' })];
    render(<PopulationNarrowingTable steps={steps} />);

    expect(screen.getAllByRole('row')).toHaveLength(3); // 1 cabecera + 2 filas
    expect(screen.getByText('Sexo: mujer')).toBeInTheDocument();
    expect(screen.getByText('Edad: 24 años')).toBeInTheDocument();
  });

  test('formatea la población restante con separador de miles en español', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ remaining_population: 24957175 })]} />);
    expect(screen.getByText('24.957.175')).toBeInTheDocument();
  });

  test('población no estimable se muestra como guion largo', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ remaining_population: null })]} />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  test.each([
    ['bajo', 'Bajo'],
    ['medio', 'Medio'],
    ['alto', 'Alto'],
    ['critico', 'Crítico'],
    ['no_estimable', 'No estimable'],
  ])('nivel de riesgo %s se etiqueta como "%s"', (risk_level, label) => {
    render(
      <PopulationNarrowingTable steps={[makeStep({ risk_level: risk_level as PopulationEstimate['risk_level'] })]} />
    );
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  test('muestra la nota inline cuando el paso tiene una', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ note: 'Nota explicativa de ejemplo' })]} />);
    expect(screen.getByText('Nota explicativa de ejemplo')).toBeInTheDocument();
  });

  test('no muestra nota inline cuando el paso no la tiene', () => {
    const { container } = render(<PopulationNarrowingTable steps={[makeStep({ note: null })]} />);
    expect(container.querySelector('.note-inline')).not.toBeInTheDocument();
  });

  test('fuente "texto" muestra el icono y etiqueta correctos', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ source: 'texto' })]} />);
    expect(screen.getByText(/✍️ Texto/)).toBeInTheDocument();
  });

  test('fuente "imagen" muestra el icono y etiqueta correctos', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ source: 'imagen' })]} />);
    expect(screen.getByText(/📷 Imagen/)).toBeInTheDocument();
  });

  test('sin filas de fuente "imagen": la nota final NO menciona las fotos', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ source: 'texto' })]} />);
    expect(screen.queryByText(/análisis visual de tus fotos/)).not.toBeInTheDocument();
  });

  test('con al menos una fila de fuente "imagen": añade la advertencia de fiabilidad', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ source: 'imagen' })]} />);
    expect(screen.getByText(/análisis visual de tus fotos/)).toBeInTheDocument();
  });

  test('siempre muestra la nota general de estimación aproximada del INE', () => {
    render(<PopulationNarrowingTable steps={[makeStep()]} />);
    expect(screen.getByText(/distribuciones agregadas del INE/)).toBeInTheDocument();
  });
});
