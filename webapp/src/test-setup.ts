// Los componentes usan useTranslation() de react-i18next, así que i18next
// tiene que estar inicializado antes de que se monte cualquier componente
// en los tests -- este import tiene el efecto secundario de llamar a
// i18n.init() (ver src/i18n/index.ts). El idioma por defecto es 'es',
// igual que los textos que los tests existentes ya esperan encontrar.
import './i18n';
