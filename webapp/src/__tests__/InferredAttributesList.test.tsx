import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { describe, expect, test } from 'vitest';
import InferredAttributesList from '../components/InferredAttributesList';
import type { InferredAttribute } from '../types';

function makeAttribute(overrides: Partial<InferredAttribute> = {}): InferredAttribute {
  return {
    category: 'aficion',
    value: 'Posible afición: senderismo',
    confidence: 0.6,
    evidence: ['https://instagram.com/p/1'],
    ...overrides,
  };
}

describe('InferredAttributesList', () => {
  test('lista vacía: no renderiza nada', () => {
    const { container } = render(<InferredAttributesList attributes={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  test('renderiza el título y subtítulo de la sección cuando hay atributos', () => {
    render(<InferredAttributesList attributes={[makeAttribute()]} />);
    expect(screen.getByText('Atributos inferidos sin correlación estadística')).toBeInTheDocument();
    expect(screen.getByText(/pueden seguir comprometiendo tu privacidad/)).toBeInTheDocument();
  });

  test('renderiza un bloque por cada atributo, con categoría traducida y valor', () => {
    const { container } = render(
      <InferredAttributesList
        attributes={[
          makeAttribute({ category: 'aficion', value: 'Posible afición: senderismo' }),
          makeAttribute({ category: 'texto_visible', value: 'Texto leído en cartel: "Bar El Rincón"' }),
        ]}
      />,
    );

    expect(container.querySelectorAll('.inferred-attribute-item')).toHaveLength(2);
    expect(screen.getByText('Afición')).toBeInTheDocument();
    expect(screen.getByText('Posible afición: senderismo')).toBeInTheDocument();
    expect(screen.getByText('Texto visible en foto')).toBeInTheDocument();
    expect(screen.getByText('Texto leído en cartel: "Bar El Rincón"')).toBeInTheDocument();
  });

  test('categoría desconocida (libre, de la IA): usa un fallback formateado a partir de la clave', () => {
    render(<InferredAttributesList attributes={[makeAttribute({ category: 'mascota_visible' })]} />);
    expect(screen.getByText('Mascota visible')).toBeInTheDocument();
  });

  test('muestra el badge de confianza aproximada cuando está presente', () => {
    render(<InferredAttributesList attributes={[makeAttribute({ confidence: 0.6 })]} />);
    expect(screen.getByText('~60%')).toBeInTheDocument();
  });

  test('muestra el número de publicaciones donde se detectó, cuando hay evidencia', () => {
    render(<InferredAttributesList attributes={[makeAttribute({ evidence: ['https://x/1', 'https://x/2'] })]} />);
    expect(screen.getByText(/Detectado en 2 publicación/)).toBeInTheDocument();
  });

  test('sin evidencia: no muestra la línea de publicaciones', () => {
    render(<InferredAttributesList attributes={[makeAttribute({ evidence: [] })]} />);
    expect(screen.queryByText(/Detectado en/)).not.toBeInTheDocument();
  });

  describe('caja informativa de la DGT (categoría "matricula")', () => {
    test('categoría "matricula": muestra la caja de la DGT con informe gratuito y de pago', () => {
      render(
        <InferredAttributesList
          attributes={[
            makeAttribute({
              category: 'matricula',
              value: 'Matrícula de vehículo visible en una foto: 1234ABC (lectura automática, puede contener errores)',
            }),
          ]}
        />,
      );

      expect(screen.getByText('Matrícula de vehículo')).toBeInTheDocument();
      expect(screen.getByText('ℹ️ Consulta oficial en la DGT')).toBeInTheDocument();
      expect(screen.getByText(/informe reducido de la DGT/)).toBeInTheDocument();
      expect(screen.getByText(/Fecha de primera matriculación en España/)).toBeInTheDocument();
      expect(screen.getAllByText(/8,67 €/).length).toBeGreaterThan(0);
      expect(screen.getByText(/Identificación del titular/)).toBeInTheDocument();
    });

    test('categoría distinta de "matricula": no muestra la caja de la DGT', () => {
      render(<InferredAttributesList attributes={[makeAttribute({ category: 'aficion' })]} />);
      expect(screen.queryByText('ℹ️ Consulta oficial en la DGT')).not.toBeInTheDocument();
    });

    test('varios atributos, solo uno de matrícula: la caja de la DGT solo aparece una vez', () => {
      const { container } = render(
        <InferredAttributesList
          attributes={[makeAttribute({ category: 'aficion' }), makeAttribute({ category: 'matricula' })]}
        />,
      );
      expect(container.querySelectorAll('.dgt-info-box')).toHaveLength(1);
    });
  });
});
