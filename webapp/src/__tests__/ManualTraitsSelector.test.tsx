import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, test, vi } from 'vitest';
import { ManualTraitsSelector } from '../components/ManualTraitsSelector';

describe('ManualTraitsSelector', () => {
  test('cerrado (isOpen=false), solo muestra el botón para abrir el panel', () => {
    render(<ManualTraitsSelector isOpen={false} setIsOpen={vi.fn()} onApplyTraits={vi.fn()} />);

    expect(
      screen.getByRole('button', { name: /Añadir rasgos físicos evidentes/ }),
    ).toBeInTheDocument();
    expect(screen.queryByText('Añadir Rasgos Físicos Evidentes')).not.toBeInTheDocument();
  });

  test('clicar el botón cerrado llama a setIsOpen(true)', async () => {
    const user = userEvent.setup();
    const setIsOpen = vi.fn();
    render(<ManualTraitsSelector isOpen={false} setIsOpen={setIsOpen} onApplyTraits={vi.fn()} />);

    await user.click(screen.getByRole('button', { name: /Añadir rasgos físicos evidentes/ }));

    expect(setIsOpen).toHaveBeenCalledWith(true);
  });

  test('abierto (isOpen=true), muestra los tres desplegables con sus opciones', () => {
    render(<ManualTraitsSelector isOpen={true} setIsOpen={vi.fn()} onApplyTraits={vi.fn()} />);

    expect(screen.getByText('Añadir Rasgos Físicos Evidentes')).toBeInTheDocument();

    const eyeSelect = screen.getByLabelText('Color de Ojos') as HTMLSelectElement;
    const hairSelect = screen.getByLabelText('Color de Pelo') as HTMLSelectElement;
    const skinSelect = screen.getByLabelText('Tono de Piel') as HTMLSelectElement;

    expect(eyeSelect).toBeInTheDocument();
    expect(hairSelect).toBeInTheDocument();
    expect(skinSelect).toBeInTheDocument();

    // Sin nada seleccionado, cada desplegable empieza en el placeholder.
    expect(eyeSelect.value).toBe('');
    expect(hairSelect.value).toBe('');
    expect(skinSelect.value).toBe('');

    // Las opciones se listan capitalizadas (ver .charAt(0).toUpperCase()).
    expect(screen.getByRole('option', { name: 'Marron' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Rubio' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Oscuro' })).toBeInTheDocument();
  });

  test('el botón X (cerrar) llama a setIsOpen(false) sin aplicar nada', async () => {
    const user = userEvent.setup();
    const setIsOpen = vi.fn();
    const onApplyTraits = vi.fn();
    render(<ManualTraitsSelector isOpen={true} setIsOpen={setIsOpen} onApplyTraits={onApplyTraits} />);

    await user.click(screen.getByRole('button', { name: 'close' }));

    expect(setIsOpen).toHaveBeenCalledWith(false);
    expect(onApplyTraits).not.toHaveBeenCalled();
  });

  test('seleccionar rasgos y pulsar "Aplicar y Recalcular" envía solo los rasgos elegidos', async () => {
    const user = userEvent.setup();
    const setIsOpen = vi.fn();
    const onApplyTraits = vi.fn();
    render(<ManualTraitsSelector isOpen={true} setIsOpen={setIsOpen} onApplyTraits={onApplyTraits} />);

    await user.selectOptions(screen.getByLabelText('Color de Ojos'), 'verde');
    await user.selectOptions(screen.getByLabelText('Tono de Piel'), 'oscuro');
    // Color de Pelo se deja sin seleccionar a propósito -- no debe aparecer
    // en el array enviado.
    await user.click(screen.getByRole('button', { name: 'Aplicar y Recalcular' }));

    expect(onApplyTraits).toHaveBeenCalledWith([
      { category: 'color_ojos', value: 'verde' },
      { category: 'color_piel', value: 'oscuro' },
    ]);
    expect(setIsOpen).toHaveBeenCalledWith(false);
  });

  test('pulsar "Aplicar y Recalcular" sin seleccionar nada envía un array vacío', async () => {
    const user = userEvent.setup();
    const onApplyTraits = vi.fn();
    render(<ManualTraitsSelector isOpen={true} setIsOpen={vi.fn()} onApplyTraits={onApplyTraits} />);

    await user.click(screen.getByRole('button', { name: 'Aplicar y Recalcular' }));

    expect(onApplyTraits).toHaveBeenCalledWith([]);
  });

  test('"Limpiar y Cerrar" envía un array vacío y cierra, aunque hubiera rasgos seleccionados', async () => {
    const user = userEvent.setup();
    const setIsOpen = vi.fn();
    const onApplyTraits = vi.fn();
    render(<ManualTraitsSelector isOpen={true} setIsOpen={setIsOpen} onApplyTraits={onApplyTraits} />);

    await user.selectOptions(screen.getByLabelText('Color de Pelo'), 'pelirrojo');
    await user.click(screen.getByRole('button', { name: 'Limpiar y Cerrar' }));

    expect(onApplyTraits).toHaveBeenCalledWith([]);
    expect(setIsOpen).toHaveBeenCalledWith(false);
  });
});
