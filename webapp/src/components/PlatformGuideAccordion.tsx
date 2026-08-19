import React, { useState, useEffect } from 'react';
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

interface Props {
  activePlatform?: string;
}

const PlatformGuideAccordion: React.FC<Props> = ({ activePlatform }) => {
  const { t } = useTranslation();
  const [selectedTab, setSelectedTab] = useState<string>(activePlatform || GUIDE_ITEMS[0].key);

  useEffect(() => {
    if (activePlatform) {
      setSelectedTab(activePlatform);
    }
  }, [activePlatform]);

  return (
    <section className="platform-guide">
      <h2>{t('landing.guide.title')}</h2>
      <p className="platform-guide-subtitle">{t('landing.guide.subtitle')}</p>

      <div className="platform-guide-tabs">
        <div className="platform-guide-tab-list" role="tablist">
          {GUIDE_ITEMS.map((item) => {
            const isActive = item.key === selectedTab;
            return (
              <details
                key={item.key}
                className={`platform-guide-item ${item.className}`}
              >
                <summary
                  id={`platform-guide-tab-${item.key}`}
                  role="tab"
                  aria-selected={isActive}
                  aria-controls={`platform-guide-content-${item.key}`}
                  tabIndex={0}
                  className={`platform-guide-tab ${item.className} ${isActive ? 'platform-guide-tab--active' : ''}`}
                  onClick={(event) => {
                    // Prevent native toggle; preserve original behaviour where
                    // clicking selects the tab but does not automatically open the <details>.
                    event.preventDefault();
                    setSelectedTab(item.key);
                  }}
                >
                  <span className="platform-guide-item-icon">{item.icon}</span>
                  <span className="platform-guide-item-name">{t(`landing.guide.${item.key}.name`)}</span>
                </summary>

                <div
                  id={`platform-guide-content-${item.key}`}
                  role="tabpanel"
                  aria-labelledby={`platform-guide-tab-${item.key}`}
                  className="platform-guide-item-content"
                >
                  {item.available ? (
                    <>
                      <h3>{t('landing.guide.reads')}</h3>
                      <ul>
                        {(t(`landing.guide.${item.key}.reads`, { returnObjects: true }) as string[]).map((line, idx) => (
                          <li key={`${item.key}-reads-${idx}`}>{line}</li>
                        ))}
                      </ul>

                      <h3>{t('landing.guide.doesNotRead')}</h3>
                      <ul>
                        {(t(`landing.guide.${item.key}.doesNotRead`, { returnObjects: true }) as string[]).map((line, idx) => (
                          <li key={`${item.key}-doesNotRead-${idx}`}>{line}</li>
                        ))}
                      </ul>

                      <h3>{t('landing.guide.todo')}</h3>
                      <ol>
                        {(t(`landing.guide.${item.key}.todo`, { returnObjects: true }) as string[]).map((line, idx) => (
                          <li key={`${item.key}-todo-${idx}`}>{line}</li>
                        ))}
                      </ol>
                    </>
                  ) : (
                    <p className="note">{t('landing.guide.unavailable')}</p>
                  )}
                </div>
              </details>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default PlatformGuideAccordion;
