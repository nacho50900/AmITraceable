import React from 'react';
import { useTranslation } from 'react-i18next';
import type { UsernameCorrelationSummary } from '../types';

interface RelatedAccountsListProps {
  relatedAccounts: UsernameCorrelationSummary | null;
}

// `related_accounts` es null cuando la comprobación está desactivada en
// este servidor (settings.enable_username_correlation, ver ADR-44/ADR-48
// y app/osint/username_correlation.py) -- en ese caso no se pinta nada,
// mismo criterio que el resto de secciones opcionales del informe
// (p.ej. LocationMap con `available=false`).
const RelatedAccountsList: React.FC<RelatedAccountsListProps> = ({ relatedAccounts }) => {
  const { t } = useTranslation();

  if (relatedAccounts === null) {
    return null;
  }

  return (
    <div className="related-accounts">
      <p className="note">
        {t('components.relatedAccounts.subtitle', { total: relatedAccounts.total_sites_checked })}
      </p>
      {relatedAccounts.matches.length === 0 ? (
        <p className="note">{t('components.relatedAccounts.noMatches')}</p>
      ) : (
        <ul className="related-accounts-list">
          {relatedAccounts.matches.map((match) => (
            <li key={match.site}>
              <a href={match.url} target="_blank" rel="noreferrer">
                {match.site}
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default RelatedAccountsList;
