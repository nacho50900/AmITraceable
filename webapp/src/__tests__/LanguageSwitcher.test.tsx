import '@testing-library/jest-dom';
import { act, fireEvent, render, screen } from '@testing-library/react';
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

  test('por defecto muestra español seleccionado, sin banderas', () => {
    render(<LanguageSwitcher />);

    const select = screen.getByLabelText('Idioma') as HTMLSelectElement;
    expect(select.value).toBe('es');
    expect(screen.getByText('Español')).toBeInTheDocument();
    expect(screen.getByText('English')).toBeInTheDocument();
    // Ningún emoji de bandera ni atributo relacionado con banderas.
    expect(screen.queryByText(/🇪🇸|🇬🇧|🇺🇸/)).not.toBeInTheDocument();
  });

  test('elegir English cambia el idioma de i18next y lo persiste en localStorage', async () => {
    render(<LanguageSwitcher />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText('Idioma'), { target: { value: 'en' } });
    });

    expect(i18n.language).toBe('en');
    expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('en');
  });

  test('tras cambiar a inglés, el propio selector se re-renderiza en inglés', async () => {
    render(<LanguageSwitcher />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText('Idioma'), { target: { value: 'en' } });
    });

    expect(screen.getByLabelText('Language')).toBeInTheDocument();
  });
});
