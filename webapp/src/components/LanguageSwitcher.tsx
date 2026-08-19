import React, { useState } from 'react';
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
  const [changing, setChanging] = useState(false);

  // i18next-browser-languagedetector puede resolver a variantes regionales
  // (p. ej. "en-US"); nos quedamos con el prefijo de dos letras para
  // marcar la opción correcta.
  const currentLang = i18n.language?.split('-')[0] === 'en' ? 'en' : 'es';

  const handleChange = async (event: React.ChangeEvent<HTMLSelectElement>) => {
    const nextLang = event.target.value as 'es' | 'en';
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
    <div className="language-switcher">
      <label htmlFor="language-selector">{t('languageSwitcher.label')}</label>
      <select
        id="language-selector"
        aria-label={t('languageSwitcher.label')}
        value={currentLang}
        onChange={handleChange}
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
