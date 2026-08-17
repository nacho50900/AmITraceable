import React from 'react';
import { useTranslation } from 'react-i18next';
import { SiInstagram, SiReddit, SiX } from 'react-icons/si';

interface GuideItem {
  key: 'reddit' | 'instagram' | 'x';
  icon: React.ReactNode;
  className: string;
  // Solo Instagram tiene contenido real por ahora -- Reddit y X muestran
  // el mismo aviso de "instrucciones no disponibles todavía" (ver
  // landing.guide.unavailable) hasta que sus integraciones funcionen.
  available: boolean;
}

const GUIDE_ITEMS: GuideItem[] = [
  { key: 'reddit', icon: <SiReddit aria-hidden="true" />, className: 'platform-guide-item--reddit', available: false },
  {
    key: 'instagram',
    icon: <SiInstagram aria-hidden="true" />,
    className: 'platform-guide-item--instagram',
    available: true,
  },
  { key: 'x', icon: <SiX aria-hidden="true" />, className: 'platform-guide-item--x', available: false },
];

const PlatformGuideAccordion: React.FC = () => {
  const { t } = useTranslation();

  return (
    <section className="platform-guide">
      <h2>{t('landing.guide.title')}</h2>
      <p className="platform-guide-subtitle">{t('landing.guide.subtitle')}</p>

      <div className="platform-guide-archive">
        {GUIDE_ITEMS.map((item) => (
          <details key={item.key} className={`platform-guide-item ${item.className}`}>
            <summary>
              <span className="platform-guide-item-icon">{item.icon}</span>
              <span className="platform-guide-item-name">{t(`landing.guide.${item.key}.name`)}</span>
              <span className="platform-guide-item-chevron" aria-hidden="true">›</span>
            </summary>

            {item.available ? (
              <div className="platform-guide-item-content">
                <h3>{t('landing.guide.reads')}</h3>
                <ul>
                  {(t(`landing.guide.${item.key}.reads`, { returnObjects: true }) as string[]).map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>

                <h3>{t('landing.guide.doesNotRead')}</h3>
                <ul>
                  {(t(`landing.guide.${item.key}.doesNotRead`, { returnObjects: true }) as string[]).map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>

                <h3>{t('landing.guide.todo')}</h3>
                <ol>
                  {(t(`landing.guide.${item.key}.todo`, { returnObjects: true }) as string[]).map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ol>
              </div>
            ) : (
              <div className="platform-guide-item-content">
                <p className="note">{t('landing.guide.unavailable')}</p>
              </div>
            )}
          </details>
        ))}
      </div>
    </section>
  );
};

export default PlatformGuideAccordion;
