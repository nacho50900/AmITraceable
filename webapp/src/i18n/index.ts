import i18n from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import { initReactI18next } from 'react-i18next';
import en from './locales/en.json';
import es from './locales/es.json';

// Clave usada tanto por el detector de idioma (para persistir la elección
// del usuario) como por el LanguageSwitcher para leer/escribir manualmente.
export const LANGUAGE_STORAGE_KEY = 'amitraceable-lang';

// 'es' es el idioma por defecto y de fallback a propósito: coincide con
// los textos que ya existían en el código antes de introducir i18n, así
// que si una clave se escribe mal o falta en algún idioma, el usuario ve
// el texto en español en vez de la clave cruda ("dashboard.title").
void i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      es: { translation: es },
      en: { translation: en },
    },
    fallbackLng: 'es',
    supportedLngs: ['es', 'en'],
    nonExplicitSupportedLngs: true,
    // Solo se detecta desde localStorage (elección explícita previa del
    // usuario vía LanguageSwitcher), NO desde el idioma del navegador: así
    // el idioma inicial es siempre 'es' (coincide con los textos
    // originales del proyecto y con lo que esperan los tests existentes)
    // hasta que el usuario elige "English" activamente.
    detection: {
      order: ['localStorage'],
      lookupLocalStorage: LANGUAGE_STORAGE_KEY,
      caches: ['localStorage'],
    },
    interpolation: {
      escapeValue: false, // React ya escapa por defecto
    },
  });

export default i18n;
