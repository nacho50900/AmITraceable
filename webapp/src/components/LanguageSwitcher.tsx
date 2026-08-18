import React, { useState, useRef, useEffect } from 'react';
import { MdLanguage, MdArrowDropDown } from 'react-icons/md';
import { useTranslation } from 'react-i18next';

// Solo dos idiomas soportados por ahora (ver src/i18n): español (por
// defecto) e inglés. Deliberadamente SIN banderas -- un idioma no es un
// país (ver, p. ej., por qué "inglés" no tiene una bandera única
// razonable), así que se usa un componente personalizado con el nombre del idioma
// en su propio idioma, precedido de un icono de globo.
const LANGUAGES = [
  { code: 'es', labelKey: 'languageSwitcher.spanish' },
  { code: 'en', labelKey: 'languageSwitcher.english' },
] as const;

const LanguageSwitcher: React.FC = () => {
  const { t, i18n } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // i18next-browser-languagedetector puede resolver a variantes regionales
  // (p. ej. "en-US"); nos quedamos con el prefijo de dos letras para
  // marcar la opción correcta.
  const currentLang = i18n.language?.split('-')[0] === 'en' ? 'en' : 'es';
  const currentLangLabel = t(LANGUAGES.find(l => l.code === currentLang)?.labelKey || 'languageSwitcher.spanish');

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="language-switcher" ref={dropdownRef}>
      <button 
        type="button" 
        className="language-switcher-button" 
        onClick={() => setIsOpen(!isOpen)}
        aria-label={t('languageSwitcher.label')}
        aria-expanded={isOpen}
      >
        <MdLanguage aria-hidden="true" className="language-switcher-icon" />
        <span className="language-switcher-current">{currentLangLabel}</span>
        <MdArrowDropDown aria-hidden="true" className="language-switcher-chevron" />
      </button>
      
      {isOpen && (
        <ul className="language-switcher-menu" role="menu">
          {LANGUAGES.map((lang) => (
            <li key={lang.code} role="none">
              <button
                role="menuitem"
                className={`language-switcher-menu-item ${currentLang === lang.code ? 'language-switcher-menu-item--active' : ''}`}
                onClick={() => {
                  i18n.changeLanguage(lang.code);
                  setIsOpen(false);
                }}
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
