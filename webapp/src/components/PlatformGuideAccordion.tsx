import React, { useEffect, useState } from 'react';
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

  const activeItem = GUIDE_ITEMS.find((item) => item.key === selectedTab) ?? GUIDE_ITEMS[0];

  return (
    <section className="platform-guide">
      <h2>{t('landing.guide.title')}</h2>
      <p className="platform-guide-subtitle">{t('landing.guide.subtitle')}</p>

      <div className="platform-guide-tabs">
        {/* Las pestañas son botones simples de tamaño fijo: no contienen el
            contenido, así que seleccionar una no altera su propio tamaño
            ni el de las demás. */}
        <div className="platform-guide-tab-list" role="tablist">
          {GUIDE_ITEMS.map((item) => {
            const isActive = item.key === selectedTab;
            return (
              <button
                key={item.key}
                type="button"
                id={`platform-guide-tab-${item.key}`}
                role="tab"
                aria-selected={isActive}
                aria-controls="platform-guide-panel"
                className={`platform-guide-tab ${item.className} ${isActive ? 'platform-guide-tab--active' : ''}`}
                onClick={() => setSelectedTab(item.key)}
              >
                <span className="platform-guide-tab-corner platform-guide-tab-corner--left" aria-hidden="true" />
                <span className="platform-guide-tab-corner-ring-wrap platform-guide-tab-corner-ring-wrap--left" aria-hidden="true">
                  <span className="platform-guide-tab-corner-circle" />
                </span>
                <span className="platform-guide-tab-border-mask platform-guide-tab-border-mask--left" aria-hidden="true" />
                <span className="platform-guide-tab-panel-border-mask" aria-hidden="true" />
                <span className="platform-guide-item-icon">{item.icon}</span>
                <span className="platform-guide-item-name">{t(`landing.guide.${item.key}.name`)}</span>
                <span className="platform-guide-tab-border-mask platform-guide-tab-border-mask--right" aria-hidden="true" />
                <span className="platform-guide-tab-corner platform-guide-tab-corner--right" aria-hidden="true" />
                <span className="platform-guide-tab-corner-ring-wrap platform-guide-tab-corner-ring-wrap--right" aria-hidden="true">
                  <span className="platform-guide-tab-corner-circle" />
                </span>
              </button>
            );
          })}
        </div>

        {/* Panel único, con fondo propio, pegado a las pestañas (border-radius
            0 en la esquina superior izquierda para que "continúe" la pestaña
            activa). */}
        <div
          id="platform-guide-panel"
          role="tabpanel"
          aria-labelledby={`platform-guide-tab-${activeItem.key}`}
          className="platform-guide-tab-content-container"
        >
          {/* Las tres plataformas se renderizan a la vez, apiladas en la
              misma celda de grid (ver .platform-guide-panel-stack): solo la
              activa es visible, pero el contenedor mide lo que ocupa la más
              alta de las tres. Así, cambiar de pestaña nunca hace crecer o
              encoger el bloque ni empuja el resto de la página -- sin
              necesidad de fijar una altura a mano (que se desincronizaría en
              cuanto cambiase el texto o el idioma). */}
          <div className="platform-guide-panel-stack">
            {GUIDE_ITEMS.map((item) => {
              const isActive = item.key === selectedTab;
              return (
                <div
                  key={item.key}
                  className="platform-guide-item-content"
                  aria-hidden={!isActive}
                  style={{ visibility: isActive ? 'visible' : 'hidden' }}
                >
                  {item.available ? (
                    <>
                      <h3>{t('landing.guide.reads')}</h3>
                      <ul>
                        {(t(`landing.guide.${item.key}.reads`, { returnObjects: true }) as string[]).map(
                          (line, idx) => (
                            <li key={`${item.key}-reads-${idx}`}>{line}</li>
                          ),
                        )}
                      </ul>

                      <h3>{t('landing.guide.doesNotRead')}</h3>
                      <ul>
                        {(t(`landing.guide.${item.key}.doesNotRead`, { returnObjects: true }) as string[]).map(
                          (line, idx) => (
                            <li key={`${item.key}-doesNotRead-${idx}`}>{line}</li>
                          ),
                        )}
                      </ul>

                      <h3>{t('landing.guide.todo')}</h3>
                      <ol>
                        {(t(`landing.guide.${item.key}.todo`, { returnObjects: true }) as string[]).map(
                          (line, idx) => (
                            <li key={`${item.key}-todo-${idx}`}>{line}</li>
                          ),
                        )}
                      </ol>
                    </>
                  ) : (
                    <p className="note">{t('landing.guide.unavailable')}</p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

export default PlatformGuideAccordion;
