import '@testing-library/jest-dom';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, test } from 'vitest';
import i18n, { LANGUAGE_STORAGE_KEY } from '../i18n';
import LanguageSwitcher from '../components/LanguageSwitcher';

describe('LanguageSwitcher', () => {
  afterEach(async () => {
    // Cada test cambia el idioma global de i18next; se restaura a 'es'
    // (el idioma por defecto del proyecto) para no dejar fugas entre tests.
    window.localStorage.removeItem(LANGUAGE_STORAGE_KEY);
    await act(async () => {
      await i18n.changeLanguage('es');
    });
  });

  test('por defecto muestra el botón en español, con Español marcado en el menú, sin banderas', async () => {
    const user = userEvent.setup();
    render(<LanguageSwitcher />);

    expect(screen.getByRole('button', { name: 'Idioma' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Idioma' }));

    const spanishOption = screen.getByRole('option', { name: 'Español' });
    expect(spanishOption).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('option', { name: 'English' })).toHaveAttribute('aria-selected', 'false');

    // Ningún emoji de bandera ni atributo relacionado con banderas.
    expect(screen.queryByText(/🇪🇸|🇬🇧|🇺🇸/)).not.toBeInTheDocument();
  });

  test('elegir English cambia el idioma de i18next y lo persiste en localStorage', async () => {
    const user = userEvent.setup();
    render(<LanguageSwitcher />);

    await user.click(screen.getByRole('button', { name: 'Idioma' }));
    await user.click(screen.getByRole('option', { name: 'English' }));

    expect(i18n.language).toBe('en');
    expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('en');
  });

  test('tras cambiar a inglés, el propio botón se re-renderiza en inglés', async () => {
    const user = userEvent.setup();
    render(<LanguageSwitcher />);

    await user.click(screen.getByRole('button', { name: 'Idioma' }));
    await user.click(screen.getByRole('option', { name: 'English' }));

    expect(screen.getByRole('button', { name: 'Language' })).toBeInTheDocument();
  });

  test('el menú se cierra tras seleccionar un idioma', async () => {
    const user = userEvent.setup();
    render(<LanguageSwitcher />);

    await user.click(screen.getByRole('button', { name: 'Idioma' }));
    expect(screen.getByRole('listbox')).toBeInTheDocument();

    await user.click(screen.getByRole('option', { name: 'English' }));

    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });
});
