# Herramienta de Análisis de Exposición de Identidad Digital (TFG)


<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=301&color=gradient&text=AmITraceable&textBg=false&fontColor=000000&section=header&reversal=false" width="100%"/>

</div>

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=tu-usuario_identity-exposure-tfg&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=tu-usuario_identity-exposure-tfg)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=tu-usuario_identity-exposure-tfg&metric=coverage)](https://sonarcloud.io/summary/new_code?id=tu-usuario_identity-exposure-tfg)

> Base de este proyecto: plantilla de laboratorio ASW (Uniovi) `yovi_en1b`.
> Se ha adaptado sustituyendo el dominio (usuarios/juego → análisis de
> exposición de identidad digital) y sustituyendo el servicio `users` de
> Node.js por un backend Python/FastAPI (necesario por las librerías de
> NLP: spaCy, sentence-transformers, scikit-learn). El servicio `gamey`
> (Rust) se ha eliminado por no aplicar a este TFG.

TFG: Análisis defensivo de huella digital propia mediante OSINT e IA.
**Versión actual limitada a Reddit** como única fuente de datos (ver
"Alcance" más abajo).

## Project Structure

- `webapp/`: frontend React + Vite + **TypeScript**. Pantalla de
  consentimiento (`Landing`) y dashboard con el informe de exposición
  (`Dashboard`).
- `users/`: backend **Python + FastAPI**. A pesar del nombre heredado de la
  plantilla, aquí vive toda la lógica de OAuth con Reddit, NLP, scoring de
  privacidad e informe (ver `users/app/`).
- `docs/`: documentación de arquitectura Arc42 (pendiente de rellenar con
  contenido específico del TFG — actualmente es la plantilla genérica).

## ⚠️ Alcance de esta versión (importante para la memoria del TFG)

- Solo Reddit. La correlación *entre plataformas* (módulo 3 de la propuesta)
  y el componente `identity_consistency_risk` del scoring quedan como
  **trabajo futuro**, documentados explícitamente en
  `users/app/scoring/privacy_score.py`.
- El usuario solo puede analizar **su propia cuenta autenticada**. No existe
  ningún flujo para analizar cuentas de terceros.
- No hay base de datos. Todo el estado vive en una cookie de sesión firmada
  (`SessionMiddleware`) con el access token de Reddit. Cerrar sesión = borrar
  todo rastro.
- Las heurísticas de inferencia de atributos
  (`users/app/nlp/attribute_inference.py`) son deliberadamente simples
  (listas de subreddits + regex) para mantener el sistema explicable.

## Componentes

### Webapp

SPA creada con [Vite](https://vitejs.dev/) y [React](https://reactjs.org/) en TypeScript.

- `src/pages/Landing.tsx`: pantalla de consentimiento + login OAuth con Reddit.
- `src/pages/Dashboard.tsx`: informe de exposición (score, atributos inferidos, gráfico horario, recomendaciones).
- `src/components/`: `ScoreBar`, `HourlyActivityChart`.
- `src/api.ts` / `src/types.ts`: cliente tipado del backend.
- `package.json`: scripts para desarrollo, tests (Vitest) y E2E (Playwright + Cucumber).

### Users (backend Python/FastAPI)

- `app/auth/reddit_oauth.py` — OAuth 2.0 con Reddit (scopes mínimos: `identity history read`).
- `app/reddit_client.py` — extracción de posts/comentarios públicos.
- `app/nlp/fingerprint.py` — fingerprinting de escritura (longitud de frase, vocabulario, emojis, patrón horario, keywords TF-IDF, idioma).
- `app/nlp/attribute_inference.py` — inferencia explicable de atributos (ubicación, ocupación, rutina) sobre la propia cuenta.
- `app/scoring/privacy_score.py` — motor de scoring de privacidad (0-100).
- `app/report/generator.py` — informe final + recomendaciones.
- `app/main.py` — app FastAPI, con métricas Prometheus en `/metrics`.
- `tests/` — pytest (unit + endpoints), con cobertura para Sonar.
- `monitoring/` — configuración de Prometheus/Grafana (heredada de la plantilla, ya apunta a `users:3000`).

## Running the Project

### With Docker

```bash
docker-compose up --build
```

- Web application: http://localhost
- Users (backend) API: http://localhost:3000 (docs interactivos en `/docs`)
- Grafana: http://localhost:9091 · Prometheus: http://localhost:9090

Antes de levantarlo, crea `users/.env` a partir de `users/.env.example` con
tus credenciales de la app de Reddit (https://www.reddit.com/prefs/apps).

### Without Docker

#### 1. Backend (`users/`)

```bash
cd users
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
cp .env.example .env   # y rellenar credenciales de Reddit
uvicorn app.main:app --reload --port 3000
```

#### 2. Webapp

```bash
cd webapp
npm install
npm run dev
```

La webapp estará en http://localhost:5173

## Available Scripts

### Webapp (`webapp/package.json`)

- `npm run dev`: servidor de desarrollo.
- `npm test`: tests unitarios (Vitest).
- `npm run test:coverage`: tests con cobertura (para Sonar).
- `npm run test:e2e`: tests E2E (levanta webapp + backend y corre Cucumber).
- `npm run start:all`: levanta webapp + backend Python a la vez (conveniencia para desarrollo/E2E).

### Users (backend Python)

- `uvicorn app.main:app --reload --port 3000`: arranca el backend en desarrollo.
- `pytest`: tests unitarios.
- `pytest --cov=app --cov-report=xml`: tests con cobertura (genera `coverage.xml` para Sonar).

## Plan de evaluación pendiente (para la memoria)

No incluido en el código, pero necesario antes de la defensa:

1. Dataset de prueba con consentimiento (cuentas propias del equipo, alts
   conocidos) para validar el módulo de inferencia de atributos.
2. Métricas: precisión de los atributos inferidos vs. verdad conocida, tasa
   de falsos positivos.
3. Comparativa de las ponderaciones del scoring (`_WEIGHTS` en
   `privacy_score.py`) contra percepción subjetiva de usuarios reales.
