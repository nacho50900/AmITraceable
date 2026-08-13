# Backend — `backend/` (Python + FastAPI)

> Renombrado desde `users/` (nombre heredado de la plantilla base del
> laboratorio ASW). Aquí vive **todo** el backend del TFG: OAuth,
> extracción de datos, NLP, k-anonimato, geolocalización por imagen,
> scoring y generación del informe.

Ver el [README raíz](../README.md) para una descripción funcional completa
del proyecto. Este documento se centra en cómo levantar y trabajar sobre
este servicio en concreto.

## Setup rápido

```bash
cd backend
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

Para que la geolocalización por imagen funcione de verdad (no solo para
poder construir el índice), instala además `requirements-vision.txt` — ver
el README raíz, sección de geolocalización.

### Recrear el entorno desde cero (Windows)

Si el venv se corrompe o quieres partir de cero:

```bat
cd C:\ruta\al\proyecto\AmITraceable\backend
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

## Datos de referencia poblacional (`app/data/ine_reference.py`)

El k-anonimato (`scoring/k_anonymity.py`) y el scoring de privacidad
necesitan saber qué tan común es cada atributo en la población española
(p. ej. "¿qué % de gente de tu edad y CCAA estudió Medicina?"). Esas
distribuciones viven en `app/data/ine_reference.py`, y se mantienen con
dos scripts en `scripts/` — herramientas de mantenimiento para refrescar
los datos de vez en cuando, no parte del pipeline de análisis en
producción.

**Comando para refrescar todo de una vez:**

```bash
cd backend
python scripts/update_ine_reference.py --insecure --apply --yes
```

`--apply` escribe los cambios en `ine_reference.py` (sin él, solo los
compara y muestra). `--yes` no pide confirmación por teclado. `--insecure`
desactiva la verificación SSL — hace falta en Windows si falla con
`CERTIFICATE_VERIFY_FAILED` (prueba antes `pip install --upgrade
certifi`, que suele arreglarlo sin necesidad de `--insecure`).

Al final, este comando llama automáticamente a
`update_studies_distribution.py` (usa `--no-studies` para saltártelo).
Hay tres flags `--force-*` (`--force-tasa-paro`, `--force-ocupacion`,
`--force-hogar`) para aplicar una tabla igualmente aunque el script avise
de que el resultado parece implausible — solo úsalos tras revisar el
aviso a mano.

### Las 8 tablas, de un vistazo

| Tabla | Fuente | Estado |
|---|---|---|
| `PROVINCE_POPULATION` | INE (Tempus3, ID 67988) | ✅ Confirmada contra la API real |
| `MARITAL_STATUS_DISTRIBUTION` / `_BY_SEX` | INE (ID 76288) | ✅ Confirmada contra la API real |
| `NATIONALITY_DISTRIBUTION` | INE (ID 59587) | ✅ Confirmada contra la API real |
| `SITUACION_LABORAL_DISTRIBUTION` | INE (IDs 65081 + 65219) | ⚠️ La tabla original (1113) resultó descontinuada desde 2013 — sustituida por dos tablas nuevas; la de paro (65219) está confirmada con datos reales hasta 2026, la de actividad (65081) es candidata de alta confianza sin confirmar todavía |
| `OCCUPATION_DISTRIBUTION` | INE (ID 65134, CNO-11) | ✅ Confirmada — corregido un bug real de doble conteo (una categoría "padre" y sus "hijos" sumaban el mismo colectivo dos veces) |
| `HOUSEHOLD_TYPE_DISTRIBUTION` | INE (PC-Axis) | ✅ Confirmada — corregidos dos bugs reales de parseo del formato |
| `STUDIES_DISTRIBUTION` | Ministerio de Universidades (Excel, sin API) | ✅ Con datos reales — método de dos pasos: total histórico por rama (egresados desde 1985) repartido según el detalle reciente por titulación |
| `LANGUAGE_BY_CCAA` | ECEPOV 2021 (INE) | ❌ Sin solución automática (encuesta puntual, sin equivalente anual) |

Cada tabla tiene su propia fecha de última verificación y umbral de
caducidad (`ine_reference.stale_tables()` dice cuáles llevan demasiado
tiempo sin refrescar). El docstring de cabecera de cada script documenta
el historial completo de investigación de cada tabla (IDs descartados,
bugs encontrados y cómo se confirmaron) — merece la pena leerlo antes de
tocar nada a mano.

## Estructura

```
app/
├── main.py                    # app FastAPI, middlewares, lifespan (precarga geolocalización), /metrics
├── config.py                  # Settings (pydantic-settings, lee .env)
├── progress.py                # callback de progreso compartido
├── analysis_router.py         # endpoints de análisis
├── ai_analysis.py             # veredicto + conclusiones sobre el informe vía Mistral AI (opcional)
├── reddit_client.py           # extracción de datos de Reddit
├── instagram_client.py        # extracción de datos de Instagram
├── auth/
│   ├── reddit_oauth.py        # OAuth 2.0 con Reddit
│   ├── instagram_oauth.py     # OAuth 2.0 con Instagram (Business Login)
│   └── dynamic_origin.py      # redirect_uri / origen del frontend derivados del Host cuando no hay valor fijo en .env
├── nlp/
│   ├── fingerprint.py         # huella de escritura
│   ├── attribute_inference.py # inferencia de atributos por comunidad/hashtag
│   ├── demographic_extraction.py  # declaraciones explícitas en texto, por regex
│   └── ai_attribute_extraction.py # lo mismo, vía IA (Mistral, opcional) -- complementa a la regex, no la sustituye
├── data/
│   ├── ine_reference.py       # tablas de distribución poblacional (INE / Ministerio de Universidades) -- ver README arriba
│   └── studies_by_university.json  # detalle completo matriculados/egresados por universidad, generado por update_studies_distribution.py
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
├── update_ine_reference.py       # refresca 6 tablas de app/data/ine_reference.py contra el INE -- ver README arriba
├── update_studies_distribution.py # refresca STUDIES_DISTRIBUTION (Ministerio de Universidades, sin API) -- ver README arriba
├── download_osv5m_spain.py    # descarga filtrada del dataset OSV-5M
├── download_osv5m_world.py    # variante sin filtro de país
├── build_faiss_index.py       # construcción del índice FAISS
└── recover_metadata.py        # reconstruye metadata.csv sin volver a descargar imágenes

tests/                         # pytest, ~153 tests, ~95% cobertura
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
| `/api/analyze/ai-summary` | POST | Envía un informe ya generado a Mistral AI, devuelve `{verdict, conclusions}` |
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
- **Módulos opcionales con degradación explícita**: la geolocalización por
  imagen (`vision/geolocation.py`), el análisis con IA (`ai_analysis.py`)
  y la extracción de atributos con IA (`nlp/ai_attribute_extraction.py`)
  están diseñados para fallar con gracia — sin índice FAISS construido (o
  sin sus dependencias pesadas instaladas, ver `requirements-vision.txt`),
  o sin `MISTRAL_API_KEY`, el resto del pipeline sigue funcionando
  exactamente igual, y el frontend distingue explícitamente "la función no
  está disponible" de "no se encontró nada" (nunca ambos mensajes a la vez).
- **Heurísticas explicables por diseño**: tanto la inferencia de atributos
  como la extracción de datos demográficos por regex usan listas + patrones
  en vez de modelos más "opacos", precisamente para que cualquier resultado
  del informe se pueda trazar hasta el post/frase concreta que lo generó
  (campo `evidence` en los modelos correspondientes). La extracción por IA
  complementa esto sin romper la trazabilidad: solo rellena huecos que la
  regex no encontró, nunca los sustituye, y una estimación de sexo por
  nombre de cuenta se marca con una procedencia distinta (`ia_nombre`) y
  menor fiabilidad que una autodeclaración real.
- **redirect_uri e origen del frontend dinámicos** (`auth/dynamic_origin.py`):
  si no hay un valor fijo en `.env`, se derivan del `Host` de la petición
  entrante. Pensado para desarrollo local con túneles rápidos de
  Cloudflare, cuya URL cambia en cada reinicio -- así no hace falta editar
  `.env` ni reiniciar el proceso cada vez, solo dar de alta la URL nueva en
  el panel de Meta (eso sí sigue siendo manual, Meta no tiene API para
  gestionar esa lista).
- **Precarga en el arranque, no en la primera petición** (`main.py`,
  `lifespan`): si la geolocalización está disponible, el modelo DINOv2 y
  el índice FAISS se cargan al arrancar el contenedor, para que el primer
  análisis de un usuario no pague ese coste.
