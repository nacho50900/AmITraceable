import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { describe, expect, test } from 'vitest';
import RelatedAccountsList from '../components/RelatedAccountsList';
import type { UsernameCorrelationSummary } from '../types';

function makeSummary(overrides: Partial<UsernameCorrelationSummary> = {}): UsernameCorrelationSummary {
  return {
    total_sites_checked: 5185,
    matches: [
      { site: 'GitHub', url: 'https://github.com/comandante', exists: true },
      { site: 'Keybase', url: 'https://keybase.io/comandante', exists: true },
    ],
    ...overrides,
  };
}

describe('RelatedAccountsList', () => {
  test('related_accounts null (funcionalidad desactivada en el backend): no renderiza nada', () => {
    const { container } = render(<RelatedAccountsList relatedAccounts={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  test('con coincidencias: muestra el total comprobado y un enlace por sitio encontrado', () => {
    render(<RelatedAccountsList relatedAccounts={makeSummary()} />);

    expect(screen.getByText(/5185/)).toBeInTheDocument();

    const github = screen.getByRole('link', { name: 'GitHub' });
    expect(github).toHaveAttribute('href', 'https://github.com/comandante');
    expect(github).toHaveAttribute('target', '_blank');
    expect(github).toHaveAttribute('rel', 'noreferrer');

    expect(screen.getByRole('link', { name: 'Keybase' })).toHaveAttribute(
      'href',
      'https://keybase.io/comandante',
    );
  });

  test('sin coincidencias: no muestra lista, muestra el mensaje de "sin resultados"', () => {
    render(<RelatedAccountsList relatedAccounts={makeSummary({ matches: [] })} />);

    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText(/No se encontró ninguna cuenta/)).toBeInTheDocument();
  });

  test('solo pinta las cuentas ENCONTRADAS -- nunca "no existe"/"no concluyente" (esas ni llegan aquí)', () => {
    render(
      <RelatedAccountsList
        relatedAccounts={makeSummary({
          matches: [{ site: 'GitHub', url: 'https://github.com/comandante', exists: true }],
        })}
      />,
    );

    expect(screen.getAllByRole('link')).toHaveLength(1);
    expect(screen.queryByText('GitLab')).not.toBeInTheDocument();
  });
});
