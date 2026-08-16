import React from 'react';
import { MdLanguage } from 'react-icons/md';
import { useTranslation } from 'react-i18next';

// Solo dos idiomas soportados por ahora (ver src/i18n): español (por
// defecto) e inglés. Deliberadamente SIN banderas -- un idioma no es un
// país (ver, p. ej., por qué "inglés" no tiene una bandera única
// razonable), así que se usa un <select> nativo con el nombre del idioma
// en su propio idioma, precedido de un icono de globo.
const LANGUAGES = [
  { code: 'es', labelKey: 'languageSwitcher.spanish' },
  { code: 'en', labelKey: 'languageSwitcher.english' },
] as const;

const LanguageSwitcher: React.FC = () => {
  const { t, i18n } = useTranslation();

  // i18next-browser-languagedetector puede resolver a variantes regionales
  // (p. ej. "en-US"); nos quedamos con el prefijo de dos letras para
  // marcar la opción correcta en el desplegable.
  const currentLang = i18n.language?.split('-')[0] === 'en' ? 'en' : 'es';

  return (
    <div className="language-switcher">
      <MdLanguage aria-hidden="true" className="language-switcher-icon" />
      <select
        className="language-switcher-select"
        value={currentLang}
        aria-label={t('languageSwitcher.label')}
        onChange={(event) => i18n.changeLanguage(event.target.value)}
      >
        {LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {t(lang.labelKey)}
          </option>
        ))}
      </select>
    </div>
  );
};

export default LanguageSwitcher;
