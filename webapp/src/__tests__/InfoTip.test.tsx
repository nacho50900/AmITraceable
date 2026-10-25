import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, test } from 'vitest';
import InfoTip from '../components/InfoTip';

function setup() {
  render(
    <div>
      <InfoTip text="Descripción de prueba" label="Más información" />
      <button type="button">Fuera</button>
    </div>,
  );
  return {
    user: userEvent.setup(),
    button: screen.getByRole('button', { name: 'Más información' }),
    bubble: screen.getByRole('tooltip'),
  };
}

describe('InfoTip', () => {
  test('empieza cerrado, con la descripción enlazada por aria-describedby', () => {
    const { button, bubble } = setup();

    expect(bubble).not.toHaveClass('info-tip-bubble--open');
    expect(button).toHaveAttribute('aria-expanded', 'false');
    expect(button).toHaveAccessibleDescription('Descripción de prueba');
  });

  test('al pasar el ratón se abre, y al salir se cierra', async () => {
    const { user, button, bubble } = setup();

    await user.hover(button);
    expect(bubble).toHaveClass('info-tip-bubble--open');
    expect(button).toHaveAttribute('aria-expanded', 'true');

    await user.unhover(button);
    expect(bubble).not.toHaveClass('info-tip-bubble--open');
  });

  test('con el teclado: el foco la abre, Escape la cierra y perder el foco también', async () => {
    const { user, bubble } = setup();

    await user.tab();
    expect(bubble).toHaveClass('info-tip-bubble--open');

    await user.keyboard('{Escape}');
    expect(bubble).not.toHaveClass('info-tip-bubble--open');

    await user.keyboard('{Tab}'); // pasa al botón "Fuera"
    await user.tab({ shift: true }); // y vuelve: reabre por foco
    expect(bubble).toHaveClass('info-tip-bubble--open');
    await user.tab();
    expect(bubble).not.toHaveClass('info-tip-bubble--open');
  });

  test('al tocar/pulsar se abre aunque no haya hover, y pulsar fuera la cierra', async () => {
    const { user, button, bubble } = setup();

    await user.pointer({ keys: '[TouchA]', target: button });
    expect(bubble).toHaveClass('info-tip-bubble--open');

    await user.click(screen.getByRole('button', { name: 'Fuera' }));
    expect(bubble).not.toHaveClass('info-tip-bubble--open');
  });
});
