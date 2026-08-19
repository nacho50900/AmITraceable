import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FiGlobe } from 'react-icons/fi';

// Solo dos idiomas soportados por ahora (ver src/i18n): español (por
// defecto) e inglés. Deliberadamente SIN banderas -- un idioma no es un
// país (ver, p. ej., por qué "inglés" no tiene una bandera única
// razonable), así que se usa un botón personalizado con un icono de
// globo seguido del nombre del idioma en su propio idioma.
const LANGUAGES = [
  { code: 'es', labelKey: 'languageSwitcher.spanish' },
  { code: 'en', labelKey: 'languageSwitcher.english' },
] as const;

const LanguageSwitcher: React.FC = () => {
  const { t, i18n } = useTranslation();
  const [changing, setChanging] = useState(false);
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // i18next-browser-languagedetector puede resolver a variantes regionales
  // (p. ej. "en-US"); nos quedamos con el prefijo de dos letras para
  // marcar la opción correcta.
  const currentLang = i18n.language?.split('-')[0] === 'en' ? 'en' : 'es';

  useEffect(() => {
    if (!open) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [open]);

  const handleSelect = async (nextLang: 'es' | 'en') => {
    setOpen(false);
    if (nextLang === currentLang) return;
    try {
      setChanging(true);
      await i18n.changeLanguage(nextLang);
    } catch (err) {
      // Log error for diagnostics; consider showing a user-visible message if desired
      // (avoid importing a UI toast here to keep the component minimal).
      // eslint-disable-next-line no-console
      console.error('Error changing language:', err);
    } finally {
      setChanging(false);
    }
  };

  return (
    <div className="language-switcher" ref={containerRef}>
      <button
        type="button"
        id="language-selector"
        className="language-switcher-button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={t('languageSwitcher.label')}
        disabled={changing}
        onClick={() => setOpen((prev) => !prev)}
      >
        <span className="language-switcher-icon" aria-hidden="true"><FiGlobe /></span>
        <span className="language-switcher-text">{t('languageSwitcher.label')}</span>
        <span className="language-switcher-chevron" aria-hidden="true">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <ul className="language-switcher-menu" role="listbox" aria-label={t('languageSwitcher.label')}>
          {LANGUAGES.map((lang) => (
            <li key={lang.code}>
              <button
                type="button"
                role="option"
                aria-selected={lang.code === currentLang}
                className={`language-switcher-menu-item ${
                  lang.code === currentLang ? 'language-switcher-menu-item--active' : ''
                }`}
                onClick={() => handleSelect(lang.code)}
              >
                {t(lang.labelKey)}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default LanguageSwitcher;
