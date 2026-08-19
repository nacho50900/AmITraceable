import '@testing-library/jest-dom';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, test } from 'vitest';
import i18n from '../i18n';
import PlatformGuideAccordion from '../components/PlatformGuideAccordion';

describe('PlatformGuideAccordion', () => {
  afterEach(async () => {
    await act(async () => {
      await i18n.changeLanguage('es');
    });
  });

  test('muestra las tres pestañas (Reddit, Instagram, X), con Reddit activa por defecto', () => {
    render(<PlatformGuideAccordion />);

    const tabs = screen.getAllByRole('tab');
    expect(tabs).toHaveLength(3);

    expect(screen.getByRole('tab', { name: 'Reddit' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Instagram' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'X' })).toBeInTheDocument();

    // Reddit es la primera pestaña del listado; es la que se muestra
    // activa por defecto al cargar (comportamiento ya existente).
    expect(screen.getByRole('tab', { name: 'Reddit' })).toHaveAttribute('aria-selected', 'true');
  });

  test('Reddit y X muestran el aviso de "no disponible todavía", no listas de lectura', async () => {
    const user = userEvent.setup();
    render(<PlatformGuideAccordion />);

    await user.click(screen.getByRole('tab', { name: 'Reddit' }));
    expect(screen.getByText(/instrucciones no disponibles todavía/i)).toBeInTheDocument();
    expect(screen.queryByRole('list')).not.toBeInTheDocument();

    await user.click(screen.getByRole('tab', { name: 'X' }));
    expect(screen.getByText(/instrucciones no disponibles todavía/i)).toBeInTheDocument();
    expect(screen.queryByRole('list')).not.toBeInTheDocument();
  });

  test('Instagram muestra las tres listas: qué lee, qué no lee, y qué debe hacer el usuario', async () => {
    const user = userEvent.setup();
    render(<PlatformGuideAccordion />);

    await user.click(screen.getByRole('tab', { name: 'Instagram' }));

    expect(screen.getByText('La aplicación SÍ lee:')).toBeInTheDocument();
    expect(screen.getByText('La aplicación NO lee:')).toBeInTheDocument();
    expect(screen.getByText('Qué tienes que hacer:')).toBeInTheDocument();

    // Contenido concreto pedido: cuenta profesional, alta en Meta for
    // Developers, contactar con el desarrollador y aceptar la invitación.
    expect(screen.getByText(/cuenta Business o Creator/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Meta for Developers/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Contacta con el desarrollador/i)).toBeInTheDocument();
    expect(screen.getByText(/Acepta la invitación/i)).toBeInTheDocument();

    // No usa reconocimiento facial: dato sensible que debe quedar explícito.
    expect(screen.getByText(/no hace reconocimiento facial/i)).toBeInTheDocument();

    const lists = screen.getAllByRole('list');
    expect(lists).toHaveLength(3); // qué lee (ul), qué no lee (ul), qué hacer (ol)
  });

  test('el tamaño de las pestañas no depende de cuál esté seleccionada', async () => {
    const user = userEvent.setup();
    render(<PlatformGuideAccordion />);

    const redditTab = screen.getByRole('tab', { name: 'Reddit' });

    // La clase base que fija padding/tamaño está siempre presente, tanto
    // seleccionada como no: el contenido vive en un panel aparte, no
    // dentro de la propia pestaña, así que su caja nunca cambia de tamaño.
    expect(redditTab).toHaveClass('platform-guide-tab');
    expect(redditTab).toHaveClass('platform-guide-tab--active');

    await user.click(screen.getByRole('tab', { name: 'X' }));

    expect(redditTab).toHaveClass('platform-guide-tab');
    expect(redditTab).not.toHaveClass('platform-guide-tab--active');
  });

  test('cambiar el idioma a inglés traduce títulos y contenido de Instagram', async () => {
    const user = userEvent.setup();
    render(<PlatformGuideAccordion />);

    await act(async () => {
      await i18n.changeLanguage('en');
    });

    expect(screen.getByText('How to use the app with each social network')).toBeInTheDocument();

    await user.click(screen.getByRole('tab', { name: 'Instagram' }));

    expect(screen.getByText('The app DOES read:')).toBeInTheDocument();
    expect(screen.getByText(/Business or Creator account/i)).toBeInTheDocument();
  });
});
