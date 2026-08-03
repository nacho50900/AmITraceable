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
    proportion: 24_957_175 / 49_128_297,
    reduction_percent: 49.2,
    ...overrides,
  };
}

describe('PopulationNarrowingTable', () => {
  test('lista vacía: muestra el mensaje explicativo, sin bloques de atributo', () => {
    const { container } = render(<PopulationNarrowingTable steps={[]} />);

    expect(screen.getByText(/No se han detectado declaraciones explícitas/)).toBeInTheDocument();
    expect(container.querySelectorAll('.population-step')).toHaveLength(0);
  });

  test('renderiza un bloque por cada paso, con su etiqueta', () => {
    const steps = [makeStep({ attribute_label: 'Sexo: mujer' }), makeStep({ attribute_label: 'Edad: 24 años' })];
    const { container } = render(<PopulationNarrowingTable steps={steps} />);

    expect(container.querySelectorAll('.population-step')).toHaveLength(2);
    expect(screen.getByText('Sexo: mujer')).toBeInTheDocument();
    expect(screen.getByText('Edad: 24 años')).toBeInTheDocument();
  });

  test('con proporción conocida: muestra el pictograma con el porcentaje', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ remaining_population: 24957175, proportion: 0.5 })]} />);
    expect(screen.getByText('50%')).toBeInTheDocument();
  });

  test('sin proporción calculable: cae al número formateado con separador de miles', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ remaining_population: 24957175, proportion: null })]} />);
    expect(screen.getByText('24.957.175')).toBeInTheDocument();
  });

  test('población no estimable (sin proporción ni población) se muestra como guion largo', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ remaining_population: null, proportion: null })]} />);
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

  test('fuente "ia" muestra el icono y etiqueta correctos', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ source: 'ia' })]} />);
    expect(screen.getByText(/🤖 IA \(texto\)/)).toBeInTheDocument();
  });

  test('fuente "ia_nombre" muestra el icono y etiqueta correctos, y añade la advertencia de fiabilidad', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ source: 'ia_nombre' })]} />);
    expect(screen.getByText(/🤖 IA \(nombre\)/)).toBeInTheDocument();
    expect(screen.getByText(/estimación por el nombre público/)).toBeInTheDocument();
  });

  test('fuente "ia_simbolica" muestra el icono y etiqueta correctos', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ source: 'ia_simbolica', category: 'estado_civil', attribute_label: 'Casado/a' })]} />);
    expect(screen.getByText(/🤖 IA \(simbólico\)/)).toBeInTheDocument();
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

  test('reduction_percent presente: muestra el badge con el porcentaje de reducción', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ reduction_percent: 49.2 })]} />);
    expect(screen.getByText('-49.2%')).toBeInTheDocument();
  });

  test('reduction_percent null (paso no estimable): no muestra el badge de reducción', () => {
    render(<PopulationNarrowingTable steps={[makeStep({ reduction_percent: null })]} />);
    expect(screen.queryByText(/^-.*%$/)).not.toBeInTheDocument();
  });
});
