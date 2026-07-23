# Backend — `users/` (Python + FastAPI)

> El nombre `users` viene de la plantilla base del laboratorio ASW y se ha
> mantenido por simplicidad en la infraestructura (docker-compose,
> monitoring), pero aquí vive **todo** el backend del TFG: OAuth,
> extracción de datos, NLP, k-anonimato, geolocalización por imagen,
> scoring y generación del informe.

Ver el [README raíz](../README.md) para una descripción funcional completa
del proyecto. Este documento se centra en cómo levantar y trabajar sobre
este servicio en concreto.

## Setup rápido

```bash
cd users
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt   # incluye requirements.txt + deps de test
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
cp .env.example .env              # y rellenar credenciales (ver abajo)
uvicorn app.main:app --reload --port 3000
```

La API queda en http://localhost:3000, con documentación interactiva
(Swagger UI) en http://localhost:3000/docs.

### Recrear el entorno desde cero (Windows)

Si el venv se corrompe o quieres partir de cero:

```bat
cd C:\ruta\al\proyecto\AmITraceable\users
rmdir /s /q venv
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

### Variables de entorno

Ver `.env.example` para la lista completa y comentada. Las obligatorias
son las de Reddit y `SESSION_SECRET_KEY`; Instagram y Mistral AI son
opcionales (sin ellas, esas funciones concretas quedan desactivadas sin
afectar al resto — ver el README raíz para el detalle).

## Estructura

```
app/
├── main.py                    # app FastAPI, middlewares, /metrics
├── config.py                  # Settings (pydantic-settings, lee .env)
├── progress.py                # callback de progreso compartido
├── analysis_router.py         # endpoints de análisis
├── ai_analysis.py             # análisis del informe vía Mistral AI (opcional)
├── reddit_client.py           # extracción de datos de Reddit
├── instagram_client.py        # extracción de datos de Instagram
├── auth/
│   ├── reddit_oauth.py        # OAuth 2.0 con Reddit
│   └── instagram_oauth.py     # OAuth 2.0 con Instagram (Business Login)
├── nlp/
│   ├── fingerprint.py         # huella de escritura
│   ├── attribute_inference.py # inferencia de atributos por comunidad/hashtag
│   └── demographic_extraction.py  # declaraciones explícitas en texto
├── data/
│   └── ine_reference.py       # tablas de distribución poblacional (INE)
├── scoring/
│   ├── privacy_score.py       # score de privacidad 0-100
│   └── k_anonymity.py         # estrechamiento de población (k-anonimato)
├── vision/
│   └── geolocation.py         # geolocalización de fotos (DINOv2+FAISS), opcional
├── report/
│   └── generator.py           # ensamblado del informe final
└── models/
    └── schemas.py             # modelos Pydantic (SocialPost, ExposureReport...)

scripts/
├── download_osv5m_spain.py    # descarga filtrada del dataset OSV-5M
├── download_osv5m_world.py    # variante sin filtro de país
└── build_faiss_index.py       # construcción del índice FAISS

tests/                         # pytest, ~117 tests, ~95% cobertura
monitoring/                    # config de Prometheus/Grafana
```

## Endpoints principales

| Ruta | Método | Descripción |
|---|---|---|
| `/auth/{reddit,instagram}/login` | GET | Redirige al proveedor OAuth |
| `/auth/{reddit,instagram}/callback` | GET | Callback OAuth, guarda tokens en sesión |
| `/auth/{reddit,instagram}/status` | GET | Estado de autenticación actual |
| `/auth/{reddit,instagram}/logout` | POST | Cierra sesión (borra la cookie) |
| `/api/analyze/{platform}` | POST | Ejecuta el pipeline completo, devuelve el informe |
| `/api/analyze/{platform}/stream` | GET | Igual que arriba, pero vía Server-Sent Events con progreso en vivo |
| `/api/analyze/ai-summary` | POST | Envía un informe ya generado a Mistral AI, devuelve conclusiones |
| `/metrics` | GET | Métricas Prometheus |
| `/docs` | GET | Swagger UI |

Todos los detalles de request/response están documentados en `/docs` una
vez arrancado el servidor (incluye los códigos de error de cada endpoint).

## Tests

```bash
pytest                                              # tests unitarios
pytest --cov=app --cov-report=xml --cov-report=term-missing   # con cobertura (para Sonar)
```

Los tests no requieren credenciales reales: usan `respx` para mockear las
llamadas HTTP a Reddit/Instagram/Mistral, y un fixture (`patch_spacy_model`
en `tests/conftest.py`) para no depender de tener el modelo de spaCy
descargado en el entorno de test.

### Advertencia silenciada: `from click.parser import split_arg_string`

Ese warning es totalmente inofensivo: no proviene de nuestro código, es una
incompatibilidad menor entre spaCy y la versión de `click` que arrastra
como dependencia (spaCy usa una API interna de `click` que va a moverse de
sitio en su versión 9.0, y `click` avisa con antelación). No afecta a nada
de lo que hace la herramienta — está silenciada explícitamente en
`pyproject.toml` (`[tool.pytest.ini_options] filterwarnings`).

## Notas de diseño relevantes para la memoria

- **Sin base de datos.** Todo el estado vive en la cookie de sesión
  firmada (`SessionMiddleware`, `same_site="none"`, `https_only=True` —
  necesario para que la sesión sobreviva peticiones cross-site cuando el
  frontend y el backend están en dominios distintos, p. ej. `localhost`
  vs. un túnel de Cloudflare).
- **Módulos opcionales con degradación explícita**: tanto la
  geolocalización por imagen (`vision/geolocation.py`) como el análisis
  con IA (`ai_analysis.py`) están diseñados para fallar con gracia — sin
  índice FAISS construido, o sin `MISTRAL_API_KEY`, el resto del pipeline
  sigue funcionando exactamente igual.
- **Heurísticas explicables por diseño**: tanto la inferencia de atributos
  como la extracción de datos demográficos usan listas + regex en vez de
  modelos más "opacos", precisamente para que cualquier resultado del
  informe se pueda trazar hasta el post/frase concreta que lo generó
  (campo `evidence` en los modelos correspondientes).
