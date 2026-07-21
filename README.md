# AmITraceable — Análisis de Exposición de Identidad Digital (TFG)

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=301&color=gradient&text=AmITraceable&textBg=false&fontColor=000000&section=header&reversal=false" width="100%"/>

</div>

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=nacho50900_Echo&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=nacho50900_Echo)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=nacho50900_Echo&metric=coverage)](https://sonarcloud.io/summary/new_code?id=nacho50900_Echo)
[![Release](https://github.com/nacho50900/AmITraceable/actions/workflows/release-deploy.yml/badge.svg)](https://github.com/nacho50900/AmITraceable/actions/workflows/release-deploy.yml)

> Base de este proyecto: plantilla de laboratorio ASW (Uniovi) `yovi_en1b`.
> Se ha adaptado sustituyendo el dominio (usuarios/juego → análisis de
> exposición de identidad digital) y sustituyendo el servicio `users` de
> Node.js por un backend Python/FastAPI (necesario por las librerías de
> NLP: spaCy, scikit-learn, DINOv2/FAISS). El servicio `gamey` (Rust) se
> ha eliminado por no aplicar a este TFG.

TFG: análisis defensivo de la propia huella digital mediante OSINT e IA. El
usuario autentica **su propia cuenta** de Reddit y/o Instagram vía OAuth, y
la herramienta genera un informe con lo que es públicamente inferible sobre
él — para que decida qué quiere seguir compartiendo, no para vigilar a
terceros.

## Qué hace la herramienta

1. **Lee tu actividad pública** (posts/comentarios de Reddit, publicaciones
   de Instagram) tras autenticarte vía OAuth.
2. **Analiza tu forma de escribir** (huella de estilo: longitud de frase,
   vocabulario, emojis, patrón horario, idioma, keywords).
3. **Infiere atributos personales** de forma explicable (ubicación,
   ocupación, rutina) a partir de en qué comunidades/hashtags participas.
4. **Detecta declaraciones explícitas** sobre ti mismo en el texto ("tengo
   24 años", "vivo en León", "estudio Medicina"...) y estima, a partir de
   distribuciones agregadas del INE, **cuánta gente en España comparte esa
   combinación de características** (estilo k-anonimato: "solo 17 personas
   en España cumplen esto").
5. **Estima la ubicación de tus fotos** (solo Instagram) comparándolas por
   similitud visual (DINOv2 + FAISS) contra un índice de imágenes
   georreferenciadas de España, mostrando los resultados en un mapa.
6. Calcula un **score de privacidad** (0-100) con desglose por componente,
   y da **recomendaciones concretas**.
7. Opcionalmente, envía el informe ya generado a un modelo de IA (Mistral
   AI, tier gratuito) para obtener conclusiones priorizadas en lenguaje
   natural.
8. Permite **descargar el informe completo en JSON** (portabilidad de
   datos, RGPD Art. 20).

Todo el pipeline corre **en memoria durante la petición**: no hay base de
datos, no se persiste nada del usuario más allá de la sesión de su propio
navegador.

## Estructura del proyecto

- `webapp/` — frontend React + Vite + **TypeScript**. Landing (consentimiento
  + login OAuth), Dashboard (informe completo con mapa, tablas y gráficos).
- `users/` — backend **Python + FastAPI**. A pesar del nombre heredado de
  la plantilla, aquí vive toda la lógica de OAuth, NLP, k-anonimato,
  geolocalización por imagen, scoring y el informe (ver `users/app/`).
- `docs/` — documentación de arquitectura Arc42.

## ⚠️ Alcance y limitaciones (importante para la memoria del TFG)

- El usuario solo puede analizar **su propia cuenta autenticada**. No existe
  ningún flujo para analizar cuentas de terceros.
- **No hay base de datos.** Todo el estado vive en una cookie de sesión
  firmada (`SessionMiddleware`) con los tokens de acceso. Cerrar sesión =
  borrar todo rastro.
- La estimación de k-anonimato (`users/app/scoring/k_anonymity.py`) usa
  **distribuciones agregadas del INE** y asume independencia entre
  atributos (no microdatos reales, no correlaciones cruzadas) — es una
  aproximación documentada, no un conteo exacto. Ver el docstring del
  módulo para la justificación de diseño frente a la alternativa
  descartada (base de datos sintética de ~49M filas).
- La geolocalización por imagen (`users/app/vision/geolocation.py`) es
  **opcional y best-effort**: si el índice FAISS no está construido (ver
  [Scripts de geolocalización](#scripts-de-geolocalizaci%C3%B3n-por-imagen-opcional)
  más abajo), esa función del pipeline simplemente no aporta nada, sin
  romper el resto del análisis. Su precisión realista es a nivel de
  provincia, no de calle (ver benchmarks de reverse geolocation citados en
  los docstrings).
- Las heurísticas de inferencia de atributos
  (`users/app/nlp/attribute_inference.py`,
  `users/app/nlp/demographic_extraction.py`) son deliberadamente simples
  (listas + regex) para mantener el sistema explicable y auditable.
- El análisis con IA (`users/app/ai_analysis.py`) es **totalmente
  opcional**: sin `MISTRAL_API_KEY` configurada, esa sección del dashboard
  simplemente indica que no está disponible; el resto de la app funciona
  igual.
- La correlación *entre plataformas* (Reddit + Instagram combinados) y el
  componente `identity_consistency_risk` del scoring quedan como
  **trabajo futuro**, documentado explícitamente en
  `users/app/scoring/privacy_score.py`.

## Componentes

### Webapp (`webapp/`)

SPA creada con [Vite](https://vitejs.dev/) y [React](https://react.dev/) en TypeScript.

- `src/pages/Landing.tsx` — pantalla de consentimiento + login OAuth (Reddit/Instagram).
- `src/pages/Dashboard.tsx` — informe completo: score, tabla de estrechamiento de población, mapa de ubicaciones estimadas, atributos inferidos, gráfico horario, perfil de escritura, recomendaciones, análisis con IA y descarga en JSON.
- `src/components/` — `ScoreBar`, `HourlyActivityChart`, `PopulationNarrowingTable`, `LocationMap` (Leaflet/OpenStreetMap), `AiSummaryCard`, `DownloadReportButton`.
- `src/api.ts` / `src/types.ts` — cliente tipado del backend.
- `src/utils/reportToJson.ts` — exportación del informe a JSON.
- Tests: Vitest + Testing Library (`src/__tests__/`), E2E con Playwright + Cucumber (`webapp/test/`, ver `webapp/E2E.md`).

### Users — backend Python/FastAPI (`users/`)

- `app/auth/reddit_oauth.py`, `app/auth/instagram_oauth.py` — OAuth 2.0 con cada plataforma.
- `app/reddit_client.py`, `app/instagram_client.py` — extracción de posts/comentarios/publicaciones públicas, normalizados a un modelo común (`SocialPost`).
- `app/nlp/fingerprint.py` — huella de escritura (longitud de frase, vocabulario, emojis, patrón horario, keywords TF-IDF, idioma).
- `app/nlp/attribute_inference.py` — inferencia explicable de atributos (ubicación, ocupación, rutina) a partir de comunidades/hashtags.
- `app/nlp/demographic_extraction.py` — extracción de declaraciones explícitas en texto (edad, sexo, ubicación, estudios, ocupación, universidad, empresa).
- `app/data/ine_reference.py` — tablas de distribución poblacional (INE) usadas para el estrechamiento de población.
- `app/scoring/k_anonymity.py` — motor de estimación de k-anonimato (estrechamiento de población en cascada).
- `app/scoring/privacy_score.py` — motor de scoring de privacidad (0-100).
- `app/vision/geolocation.py` — geolocalización de fotos por similitud visual (DINOv2 + FAISS), opcional.
- `app/ai_analysis.py` — análisis del informe vía Mistral AI, opcional.
- `app/progress.py` — callback de progreso compartido, usado por el endpoint de streaming.
- `app/analysis_router.py` — endpoints de análisis (`/api/analyze/{platform}`, `/api/analyze/{platform}/stream`, `/api/analyze/ai-summary`).
- `app/report/generator.py` — ensamblado del informe final + recomendaciones.
- `app/main.py` — app FastAPI, métricas Prometheus en `/metrics`.
- `tests/` — pytest (unit + endpoints), ~95% cobertura, para Sonar.
- `scripts/` — descarga del dataset OSV-5M y construcción del índice FAISS (ver más abajo).
- `monitoring/` — configuración de Prometheus/Grafana.

## Running the Project

### With Docker

```bash
docker-compose up --build
```

- Web application: http://localhost
- Users (backend) API: http://localhost:3000 (docs interactivos en `/docs`)
- Grafana: http://localhost:9091 · Prometheus: http://localhost:9090

Antes de levantarlo, crea `users/.env` a partir de `users/.env.example`
(ver [variables de entorno](#variables-de-entorno) más abajo).

### Without Docker

#### 1. Backend (`users/`)

```bash
cd users
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
cp .env.example .env              # y rellenar credenciales
uvicorn app.main:app --reload --port 3000
```

En arranques posteriores, con el venv ya creado, basta con activar el
entorno y lanzar uvicorn directamente.

#### 2. Webapp

```bash
cd webapp
npm install
npm run dev
```

La webapp estará en http://localhost:5173

### Variables de entorno

Ver `users/.env.example` para la lista completa comentada. Resumen:

| Variable | Obligatoria | Notas |
|---|---|---|
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_REDIRECT_URI`, `REDDIT_USER_AGENT` | Sí | App tipo "web app" en https://www.reddit.com/prefs/apps |
| `SESSION_SECRET_KEY` | Sí | Cadena aleatoria larga, firma la cookie de sesión |
| `INSTAGRAM_APP_ID`, `INSTAGRAM_APP_SECRET`, `INSTAGRAM_REDIRECT_URI` | No | Sin ellas, Instagram devuelve 503 pero Reddit sigue funcionando. App "API setup with Instagram Login" en Meta for Developers; requiere `redirect_uri` HTTPS (ver nota de túnel más abajo) |
| `FRONTEND_ORIGIN` | Sí | Para CORS; por defecto `http://localhost:5173` |
| `MISTRAL_API_KEY` | No | Tier gratuito de [Mistral AI](https://console.mistral.ai). Sin ella, la sección "Analizar con IA" del dashboard indica que no está disponible, sin afectar al resto |
| `MISTRAL_MODEL` | No | Por defecto `mistral-small-latest` |

**Nota sobre Instagram y HTTPS en local:** la API de Instagram (Business
Login) exige que `redirect_uri` sea HTTPS, incluso en desarrollo. Para
probarlo en local sin dominio propio, usa un túnel de Cloudflare:

```bash
cloudflared tunnel --url http://localhost:3000
```

y usa la URL `https://xxx.trycloudflare.com` que te dé tanto en
`INSTAGRAM_REDIRECT_URI` como en la app de Meta Developers (con el path
`/auth/instagram/callback`), y en `VITE_API_URL` de la webapp para que el
login y el callback compartan dominio (la cookie de sesión no viaja entre
`localhost` y el túnel).

### Scripts de geolocalización por imagen (opcional)

El módulo de geolocalización de fotos (`app/vision/geolocation.py`) es
opcional: sin el índice FAISS construido, simplemente no aporta nada al
informe, sin errores. Para activarlo:

```bash
cd users
pip install torch transformers faiss-cpu huggingface_hub pandas pillow tqdm

python scripts/download_osv5m_spain.py --output data/osv5m_spain --max-disk-gb 35
python scripts/build_faiss_index.py --images data/osv5m_spain
```

- `download_osv5m_spain.py` descarga solo las imágenes de España del
  dataset [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m)
  (streaming shard a shard, con límite de disco configurable, reanudable
  tras interrupción).
- `download_osv5m_world.py` es la variante sin filtro de país (mucho más
  pesada en tráfico de red, ~260GB).
- `build_faiss_index.py` extrae embeddings con DINOv2 y construye el
  índice de búsqueda por similitud.

Estos datos/artefactos **no se versionan** en el repositorio (ver
`.gitignore`): son regenerables ejecutando los scripts.

## Available Scripts

### Webapp (`webapp/package.json`)

- `npm run dev` — servidor de desarrollo.
- `npm test` — tests unitarios (Vitest).
- `npm run test:coverage` — tests con cobertura (para Sonar).
- `npm run test:e2e` — tests E2E (levanta webapp + backend y corre Cucumber; ver `webapp/E2E.md`).
- `npm run start:all` — levanta webapp + backend Python a la vez (conveniencia para desarrollo/E2E).
- `npm run lint` — ESLint.

### Users (backend Python)

- `uvicorn app.main:app --reload --port 3000` — arranca el backend en desarrollo.
- `pytest` — tests unitarios (~117 tests).
- `pytest --cov=app --cov-report=xml --cov-report=term` — tests con cobertura (genera `coverage.xml` para Sonar).

## Respecto a SonarQube

Debido a restricciones de la cuenta gratuita, el token de Sonar debe
renovarse cada dos meses. Configuración en `sonar-project.properties`
(rutas de fuentes/tests, exclusiones, reportes de cobertura).

## Plan de evaluación pendiente (para la memoria)

No incluido en el código, pero necesario antes de la defensa:

1. Dataset de prueba con consentimiento (cuentas propias del equipo, alts
   conocidos) para validar el módulo de inferencia de atributos y de
   geolocalización por imagen.
2. Métricas: precisión de los atributos inferidos vs. verdad conocida, tasa
   de falsos positivos, precisión real del módulo de geolocalización sobre
   fotos propias geoetiquetadas.
3. Comparativa de las ponderaciones del scoring (`_WEIGHTS` en
   `privacy_score.py`) y de la asunción de independencia del estrechamiento
   de población (`k_anonymity.py`) contra percepción subjetiva de usuarios
   reales o correlaciones reales del Censo/EPA.
