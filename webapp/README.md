# Webapp — `webapp/` (React + Vite + TypeScript)

Frontend del TFG **AmITraceable**. Ver el [README raíz](../README.md) para
una descripción funcional completa del proyecto.

## Setup rápido

```bash
cd webapp
npm install
npm run dev
```

La webapp queda en http://localhost:5173. Necesita el backend (`users/`)
corriendo en http://localhost:3000 (o la URL que apuntes en
`VITE_API_URL`, ver más abajo).

### Variable de entorno `VITE_API_URL`

Por defecto la webapp llama al backend en `http://localhost:3000`. Si el
backend está en otra URL (por ejemplo, un túnel de Cloudflare al probar el
login de Instagram en local, ver el README raíz), crea `webapp/.env` o
`webapp/.env.local`:

```
VITE_API_URL=https://tu-url-del-backend
```

Reinicia `npm run dev` tras cambiarla (Vite solo lee `.env` al arrancar).

## Estructura

```
src/
├── pages/
│   ├── Landing.tsx             # consentimiento + login OAuth
│   └── Dashboard.tsx           # informe completo
├── components/
│   ├── ScoreBar.tsx
│   ├── HourlyActivityChart.tsx
│   ├── PopulationNarrowingTable.tsx   # tabla de estrechamiento de población
│   ├── LocationMap.tsx                # mapa de ubicaciones estimadas (Leaflet)
│   ├── AiSummaryCard.tsx              # análisis con IA (opcional)
│   └── DownloadReportButton.tsx       # exportar informe a JSON
├── utils/
│   └── reportToJson.ts         # serialización del informe a JSON
├── api.ts                      # cliente tipado del backend
├── types.ts                    # tipos compartidos con el backend
└── __tests__/                  # tests unitarios (Vitest + Testing Library)

test/                           # tests E2E (Playwright + Cucumber), ver E2E.md
```

## Available Scripts

- `npm run dev` — servidor de desarrollo con hot reload.
- `npm run build` — build de producción (`tsc -b && vite build`).
- `npm run lint` — ESLint.
- `npm test` — tests unitarios (Vitest).
- `npm run test:coverage` — tests con cobertura (para Sonar).
- `npm run test:watch` — tests en modo watch.
- `npm run test:e2e` — tests E2E completos (levanta webapp + backend y corre Cucumber). Ver [`E2E.md`](./E2E.md) para más detalle.
- `npm run test:e2e:install-browsers` — instala los navegadores de Playwright (una vez).
- `npm run start:all` — levanta webapp + backend Python a la vez (conveniencia para desarrollo/E2E).

## Notas técnicas

- **`LocationMap`** usa [react-leaflet](https://react-leaflet.js.org/) +
  OpenStreetMap (gratis, sin API key). En los tests unitarios, `react-leaflet`
  se mockea porque depende de APIs de navegador real (mediciones de DOM)
  que `jsdom` no implementa de forma fiable — ver `LocationMap.test.tsx` y
  `Dashboard.test.tsx` para el patrón de mock usado.
- **`HourlyActivityChart`** usa [Recharts](https://recharts.org/), que
  necesita `ResizeObserver`; en los tests que renderizan `Dashboard`
  completo hay un polyfill mínimo para eso.
- **Descarga de informes**: `downloadReportAsJson` genera el `Blob` en el
  propio navegador, sin ninguna petición al servidor — el informe ya está
  en memoria tras el análisis.
