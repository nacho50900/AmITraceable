import '@testing-library/jest-dom';
import { act, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, test } from 'vitest';
import i18n from '../i18n';
import PlatformGuideAccordion from '../components/PlatformGuideAccordion';

describe('PlatformGuideAccordion', () => {
  afterEach(async () => {
    await act(async () => {
      await i18n.changeLanguage('es');
    });
  });

  test('muestra las tres pestañas (Reddit, Instagram, X), todas cerradas por defecto', () => {
    const { container } = render(<PlatformGuideAccordion />);

    const items = container.querySelectorAll('.platform-guide-item');
    expect(items).toHaveLength(3);
    items.forEach((item) => expect(item).not.toHaveAttribute('open'));

    expect(screen.getByText('Reddit')).toBeInTheDocument();
    expect(screen.getByText('Instagram')).toBeInTheDocument();
    expect(screen.getByText('X')).toBeInTheDocument();
  });

  test('Reddit y X muestran el aviso de "no disponible todavía", no listas de lectura', () => {
    const { container } = render(<PlatformGuideAccordion />);

    const reddit = container.querySelector('.platform-guide-item--reddit') as HTMLElement;
    const x = container.querySelector('.platform-guide-item--x') as HTMLElement;

    expect(within(reddit).getByText(/instrucciones no disponibles todavía/i)).toBeInTheDocument();
    expect(within(x).getByText(/instrucciones no disponibles todavía/i)).toBeInTheDocument();
    expect(within(reddit).queryByRole('list')).not.toBeInTheDocument();
  });

  test('Instagram muestra las tres listas: qué lee, qué no lee, y qué debe hacer el usuario', () => {
    const { container } = render(<PlatformGuideAccordion />);

    const instagram = container.querySelector('.platform-guide-item--instagram') as HTMLElement;
    const withinInstagram = within(instagram);

    expect(withinInstagram.getByText('La aplicación SÍ lee:')).toBeInTheDocument();
    expect(withinInstagram.getByText('La aplicación NO lee:')).toBeInTheDocument();
    expect(withinInstagram.getByText('Qué tienes que hacer:')).toBeInTheDocument();

    // Contenido concreto pedido: cuenta profesional, alta en Meta for
    // Developers, contactar con el desarrollador y aceptar la invitación.
    expect(withinInstagram.getByText(/cuenta Business o Creator/i)).toBeInTheDocument();
    expect(withinInstagram.getAllByText(/Meta for Developers/i).length).toBeGreaterThan(0);
    expect(withinInstagram.getByText(/Contacta con el desarrollador/i)).toBeInTheDocument();
    expect(withinInstagram.getByText(/Acepta la invitación/i)).toBeInTheDocument();

    // No usa reconocimiento facial: dato sensible que debe quedar explícito.
    expect(withinInstagram.getByText(/no hace reconocimiento facial/i)).toBeInTheDocument();

    const lists = withinInstagram.getAllByRole('list');
    expect(lists).toHaveLength(3); // qué lee (ul), qué no lee (ul), qué hacer (ol)
  });

  test('cambiar el idioma a inglés traduce títulos y contenido de Instagram', async () => {
    render(<PlatformGuideAccordion />);

    await act(async () => {
      await i18n.changeLanguage('en');
    });

    expect(screen.getByText('How to use the app with each social network')).toBeInTheDocument();
    expect(screen.getByText('The app DOES read:')).toBeInTheDocument();
    expect(screen.getByText(/Business or Creator account/i)).toBeInTheDocument();
  });
});
