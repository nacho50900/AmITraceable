import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, test, vi } from 'vitest';
import '../i18n';
import { ManualTraitsSelector } from '../components/ManualTraitsSelector';

describe('ManualTraitsSelector', () => {
  test('cerrado (isOpen=false), solo muestra el botón para abrir el panel', () => {
    render(<ManualTraitsSelector isOpen={false} setIsOpen={vi.fn()} onApplyTraits={vi.fn()} />);

    expect(
      screen.getByRole('button', { name: /Añadir rasgos físicos evidentes/ }),
    ).toBeInTheDocument();
    expect(screen.queryByText('Añadir rasgos físicos evidentes', { selector: 'h3' })).not.toBeInTheDocument();
  });

  test('clicar el botón cerrado llama a setIsOpen(true)', async () => {
    const user = userEvent.setup();
    const setIsOpen = vi.fn();
    render(<ManualTraitsSelector isOpen={false} setIsOpen={setIsOpen} onApplyTraits={vi.fn()} />);

    await user.click(screen.getByRole('button', { name: /Añadir rasgos físicos evidentes/ }));

    expect(setIsOpen).toHaveBeenCalledWith(true);
  });

  test('abierto (isOpen=true), muestra los tres desplegables con sus opciones reales (3/4/3)', () => {
    render(<ManualTraitsSelector isOpen={true} setIsOpen={vi.fn()} onApplyTraits={vi.fn()} />);

    expect(screen.getByRole('heading', { name: /Añadir rasgos físicos evidentes/ })).toBeInTheDocument();

    const eyeSelect = screen.getByLabelText('Color de ojos') as HTMLSelectElement;
    const hairSelect = screen.getByLabelText('Color de pelo') as HTMLSelectElement;
    const skinSelect = screen.getByLabelText('Tono de piel') as HTMLSelectElement;

    expect(eyeSelect).toBeInTheDocument();
    expect(hairSelect).toBeInTheDocument();
    expect(skinSelect).toBeInTheDocument();

    // Sin nada seleccionado, cada desplegable empieza en el placeholder.
    expect(eyeSelect.value).toBe('');
    expect(hairSelect.value).toBe('');
    expect(skinSelect.value).toBe('');

    // Categorías respaldadas por Navarro-Lopez et al. (Genes 2024): 3
    // valores de ojos, 4 de pelo, 3 de piel -- no las 5/5/3 inventadas
    // originales.
    expect(screen.getByRole('option', { name: 'Marrón' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Intermedio' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Azul' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Castaño' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Negro' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Rubio' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Pelirrojo' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Claro' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Medio' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Oscuro' })).toBeInTheDocument();

    // Ninguna opción inventada de la versión original (verde/miel/moreno/
    // canoso) debe seguir presente.
    expect(screen.queryByRole('option', { name: 'Verde' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Miel' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Moreno' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Canoso' })).not.toBeInTheDocument();
  });

  test('el botón X (cerrar) llama a setIsOpen(false) sin aplicar nada', async () => {
    const user = userEvent.setup();
    const setIsOpen = vi.fn();
    const onApplyTraits = vi.fn();
    render(<ManualTraitsSelector isOpen={true} setIsOpen={setIsOpen} onApplyTraits={onApplyTraits} />);

    await user.click(screen.getByRole('button', { name: 'Cerrar' }));

    expect(setIsOpen).toHaveBeenCalledWith(false);
    expect(onApplyTraits).not.toHaveBeenCalled();
  });

  test('seleccionar rasgos y pulsar "Aplicar y recalcular" envía solo los rasgos elegidos', async () => {
    const user = userEvent.setup();
    const setIsOpen = vi.fn();
    const onApplyTraits = vi.fn();
    render(<ManualTraitsSelector isOpen={true} setIsOpen={setIsOpen} onApplyTraits={onApplyTraits} />);

    await user.selectOptions(screen.getByLabelText('Color de ojos'), 'azul');
    await user.selectOptions(screen.getByLabelText('Tono de piel'), 'oscuro');
    // Color de pelo se deja sin seleccionar a propósito -- no debe aparecer
    // en el array enviado.
    await user.click(screen.getByRole('button', { name: 'Aplicar y recalcular' }));

    expect(onApplyTraits).toHaveBeenCalledWith([
      { category: 'color_ojos', value: 'azul' },
      { category: 'color_piel', value: 'oscuro' },
    ]);
    expect(setIsOpen).toHaveBeenCalledWith(false);
  });

  test('pulsar "Aplicar y recalcular" sin seleccionar nada envía un array vacío', async () => {
    const user = userEvent.setup();
    const onApplyTraits = vi.fn();
    render(<ManualTraitsSelector isOpen={true} setIsOpen={vi.fn()} onApplyTraits={onApplyTraits} />);

    await user.click(screen.getByRole('button', { name: 'Aplicar y recalcular' }));

    expect(onApplyTraits).toHaveBeenCalledWith([]);
  });

  test('"Limpiar y cerrar" envía un array vacío y cierra, aunque hubiera rasgos seleccionados', async () => {
    const user = userEvent.setup();
    const setIsOpen = vi.fn();
    const onApplyTraits = vi.fn();
    render(<ManualTraitsSelector isOpen={true} setIsOpen={setIsOpen} onApplyTraits={onApplyTraits} />);

    await user.selectOptions(screen.getByLabelText('Color de pelo'), 'pelirrojo');
    await user.click(screen.getByRole('button', { name: 'Limpiar y cerrar' }));

    expect(onApplyTraits).toHaveBeenCalledWith([]);
    expect(setIsOpen).toHaveBeenCalledWith(false);
  });
});
