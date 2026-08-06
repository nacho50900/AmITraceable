# Graph Report - .  (2026-08-06)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1093 nodes · 2494 edges · 94 communities (57 shown, 37 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 152 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6fd143c4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 93

## God Nodes (most connected - your core abstractions)
1. `DemographicFindings` - 133 edges
2. `SocialPost` - 85 edges
3. `extract_demographics()` - 73 edges
4. `estimate_population_narrowing()` - 57 edges
5. `_post()` - 52 edges
6. `generate_report()` - 51 edges
7. `extract_demographics_with_ai()` - 48 edges
8. `_post()` - 41 edges
9. `_fingerprint()` - 40 edges
10. `_score()` - 40 edges

## Surprising Connections (you probably didn't know these)
- `TestAnalyzeReportWithAi` --uses--> `AiAnalysisUnavailable`  [INFERRED]
  backend/tests/test_ai_analysis.py → backend/app/ai_analysis.py
- `InstagramClient` --uses--> `SocialPost`  [INFERRED]
  backend/app/instagram_client.py → backend/app/models/schemas.py
- `InstagramClient` --uses--> `SocialProfile`  [INFERRED]
  backend/app/instagram_client.py → backend/app/models/schemas.py
- `AiExtractionUnavailable` --uses--> `SocialPost`  [INFERRED]
  backend/app/nlp/ai_attribute_extraction.py → backend/app/models/schemas.py
- `DemographicFindings` --uses--> `SocialPost`  [INFERRED]
  backend/app/nlp/demographic_extraction.py → backend/app/models/schemas.py

## Import Cycles
- None detected.

## Communities (94 total, 37 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (35): InferredAttribute, _build_recommendations(), generate_report(), ExposureReport, InferredAttribute, PrivacyScore, WritingFingerprint, _fingerprint() (+27 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (21): extract_demographics_with_ai(), Punto de entrada del módulo. Nunca lanza excepciones: cualquier fallo (sin API…, _mock_content(), _post(), asyncio, fixture, Si el modelo (por error o porque el texto lo permitía) devuelve ambos campos,…, Si el LLM no devuelve el campo (o devuelve algo con forma inesperada), no debe… (+13 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (33): age_bin(), _build_age_distribution_1y(), _build_ccaa_population(), Tablas de distribución poblacional agregadas, usadas por…, Convierte una edad concreta en su tramo quinquenal de AGE_DISTRIBUTION_5Y. Se…, Deriva una proporción por EDAD EXACTA (año a año) a partir de…, Normaliza `raw_region` (tal como viene de geolocation.py, en español o inglés,…, Suma la población de las provincias de cada comunidad autónoma a partir de… (+25 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (23): AiAnalysisUnavailable, Exception, Módulo 8 (nuevo, opcional): pide a un LLM (Mistral AI) que lea el informe de…, Se lanza cuando el análisis con IA no se puede realizar (sin API key…, ExposureReport, PrivacyScore, SocialProfile, WritingFingerprint (+15 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (16): _FakeModel, _install_fake_model(), fixture, _RaisingModel, Tests de app/vision/scene_analysis.py. No se descarga el modelo real (~1.8B…, Sustituye a Moondream2 lo justo para analyze_image_content: solo necesita…, No es un test de comportamiento del modelo real (eso no se puede probar aquí…, Cada test debe partir de _model limpio, igual que geolocation.py. (+8 more)

### Community 5 - "Community 5"
Cohesion: 0.16
Nodes (6): DemographicFindings, estimate_population_narrowing(), Reproduce el ejemplo de la conversación: mujer, 24 años, vive en León, estudia…, El segundo escalón debe reducir respecto a lo que quedaba TRAS el primero, no…, TestEstimatePopulationNarrowing, TestNacionalidadStep

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (31): leaflet, react, react-dom, react-icons, react-leaflet, react-router-dom, recharts, dependencies (+23 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (26): ai_summary(), analyze(), analyze_stream(), _build_report(), _instagram_client_from_session(), ExposureReport, get, post (+18 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (26): SocialPost, infer_attributes(), _infer_location(), _infer_occupation(), _infer_routine(), _matches(), InferredAttribute, Inferencia de atributos personales a partir del contenido público del propio… (+18 more)

### Community 9 - "Community 9"
Cohesion: 0.10
Nodes (27): ImageLocationPoint, _apply_ai_findings(), _apply_home_candidate(), _apply_image_geolocation(), _filter_and_resolve_estimates(), _group_by_ccaa(), _HomeRegionCandidate, _infer_home_region() (+19 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (16): InstagramClient, AsyncClient, ProgressCallback, Todas las URLs de foto analizables de una publicación. En un carrusel, la API…, Pide también `name`, `biography` y `profile_picture_url`, usados…, mock_instagram_api(), asyncio, fixture (+8 more)

### Community 11 - "Community 11"
Cohesion: 0.07
Nodes (26): DOM, DOM.Iterable, ES2023, node, src, vite/client, vite.config.ts, compilerOptions (+18 more)

### Community 12 - "Community 12"
Cohesion: 0.28
Nodes (8): analyze_report_with_ai(), ExposureReport, _make_report(), _mock_content(), asyncio, ExposureReport, report.recommendations ya no se muestra como sección propia en el dashboard,…, TestAnalyzeReportWithAi

### Community 13 - "Community 13"
Cohesion: 0.19
Nodes (22): _apply_proportion(), _location_no_estimable_step(), _location_note(), PopulationNarrowingStep, Módulo 6 (nuevo): estima cuánta gente en España comparte cada combinación de…, Multiplica `remaining` por una proporción marginal del INE (asumiendo…, Usa municipio si está disponible (más específico); si no, provincia; si no,…, La tabla LANGUAGE_BY_CCAA está condicionada a la comunidad autónoma (ver… (+14 more)

### Community 14 - "Community 14"
Cohesion: 0.16
Nodes (21): Como `resolve_autonomous_community`, pero para una frase libre (p.ej.…, resolve_autonomous_community_in_text(), _detect_household_type(), extract_demographics(), _mark_all_detected_as_texto(), _match_location(), Extracción de datos demográficos AUTODECLARADOS por el propio usuario en su…, Todo lo detectado por este módulo viene de texto autodeclarado (por definición:… (+13 more)

### Community 15 - "Community 15"
Cohesion: 0.20
Nodes (11): _FakeImage, _install_fake_embedding(), _install_fake_index(), DataFrame, Evita depender de torch/transformers reales (no instalados en este entorno,…, Caso motivador: una foto de solo mar/cielo/primer plano puede parecerse…, Con menos de _MIN_NEIGHBORS_WITH_COORDS_FOR_SPREAD_CHECK vecinos con…, Si la foto trae GPS real en el EXIF, se usa directamente (la región conocida… (+3 more)

### Community 16 - "Community 16"
Cohesion: 0.17
Nodes (17): current_origin(), frontend_target(), Request, Deriva dinámicamente el origen (esquema + host) de la petición entrante.…, Esquema+host de la petición actual, asumiendo siempre HTTPS (todo este proyecto…, URL base a la que redirigir tras un login (correcto o denegado). Prioridad:…, callback(), login() (+9 more)

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (4): _post(), TestAge, TestHouseholdType, TestSex

### Community 18 - "Community 18"
Cohesion: 0.16
Nodes (9): asyncio, El caso pedido: un carrusel con varias fotos debe analizarlas TODAS, no solo la…, El caso pedido: el análisis de contenido visual (aficiones, señal de pareja --…, Sin descripción (modelo no disponible, o falló para esta foto en concreto), el…, Ya no se descarta nada por umbral aquí dentro -- eso es responsabilidad de…, Regresión: `estimate_location_from_image` es síncrona y hace trabajo de CPU…, Regresión: con `Settings.photo_analysis_concurrency >= 2`, la foto 2 debe…, Sin índice/dependencias, ni se intenta descargar nada -- y se distingue… (+1 more)

### Community 19 - "Community 19"
Cohesion: 0.40
Nodes (6): AiSummaryCardProps, DownloadReportButton(), DownloadReportButtonProps, makeExposureReport(), ExposureReport, downloadReportAsJson()

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (7): mock, Tests del módulo de autenticación con Instagram. Las llamadas HTTP reales a…, Sin INSTAGRAM_REDIRECT_URI en el entorno, el redirect_uri se deriva del Host de…, Mismo fallback que en Reddit (ver app/auth/dynamic_origin.py), para que ambas…, test_callback_falls_back_to_request_host_without_frontend_origin(), test_instagram_callback_exchanges_tokens_and_sets_session(), TestDynamicRedirectUriFallback

### Community 21 - "Community 21"
Cohesion: 0.16
Nodes (12): formatPopulation(), PopulationNarrowingTable(), PopulationNarrowingTableProps, RISK_COLORS, RISK_LABELS, SOURCE_ICONS, SOURCE_LABELS, SOURCE_TITLES (+4 more)

### Community 22 - "Community 22"
Cohesion: 0.12
Nodes (7): El paso pedido explícitamente: la inferencia simbólica de IA sobre el estado…, Si la IA no encontró ninguna señal en ningún sentido, no debe inventarse un…, Las proporciones de MARITAL_STATUS_DISTRIBUTION deben sumar 1.0 -- si no,…, Debe estrechar MÁS que sexo solo, no sustituirlo ni ser independiente de él --…, Es una categoría ENCADENABLE (_CHAINED_CATEGORIES): debe poder ser, ella sola,…, Cuando también se conoce el sexo, debe usarse la proporción EXACTA de esa…, TestEstadoCivilStep

### Community 23 - "Community 23"
Cohesion: 0.23
Nodes (9): AsyncClient, ProgressCallback, RedditClient, mock_reddit_api(), asyncio, fixture, Tests del cliente de Reddit: extracción de posts/comentarios y su normalización…, test_fetch_profile_feeds_the_common_pipeline() (+1 more)

### Community 24 - "Community 24"
Cohesion: 0.14
Nodes (9): _FakeTensor, _make_fake_torch(), _NoGradContext, fixture, Tests de app/vision/geolocation.py. No se descarga ningún modelo real ni se…, Sustituye a torch.Tensor lo justo para lo que geolocation.py necesita:…, torch es una dependencia OPCIONAL de este proyecto (solo la necesita el módulo…, Cada test debe partir de _model/_processor/_index/_index_meta limpios, para que… (+1 more)

### Community 25 - "Community 25"
Cohesion: 0.26
Nodes (13): callback(), login(), logout(), get, post, Request, Módulo de autenticación con Instagram, vía "Business Login for Instagram".…, Borra solo las claves de Instagram de la sesión (no afecta a Reddit, ya que… (+5 more)

### Community 26 - "Community 26"
Cohesion: 0.25
Nodes (7): PopulationEstimate, Modelos de datos. Todo esto vive en memoria durante la petición HTTP, nunca se…, detect_travel_permalinks(), Detección heurística (regex) de publicaciones cuyo pie de foto indica que la…, Devuelve el conjunto de permalinks de `posts` cuyo texto (caption) contiene…, _post(), TestDetectTravelPermalinks

### Community 27 - "Community 27"
Cohesion: 0.23
Nodes (9): ChartPoint, HourlyActivityChart(), HourlyActivityChartProps, riskLabel(), ScoreBar(), ScoreBarProps, Dashboard(), formatPhotosLabel() (+1 more)

### Community 28 - "Community 28"
Cohesion: 0.22
Nodes (10): confidenceColor(), formatDate(), LocationMap(), LocationMapProps, SPAIN_CENTER, AuthStatus, ImageLocationPoint, InferredAttribute (+2 more)

### Community 29 - "Community 29"
Cohesion: 0.24
Nodes (12): _estimate_from_exact_coordinates(), estimate_location_from_image(), _extract_exif_gps(), GeolocationOutcome, _haversine_km(), ImageLocationEstimate, _neighbor_spread_km(), Módulo 7 (nuevo): estima la provincia/ciudad más probable de una imagen… (+4 more)

### Community 30 - "Community 30"
Cohesion: 0.28
Nodes (12): _dir_size_gb(), _download_and_extract_shard(), _is_spain(), _load_spain_ids(), main(), DataFrame, Lock, Path (+4 more)

### Community 31 - "Community 31"
Cohesion: 0.26
Nodes (11): analyze_image_content(), _lazy_load(), _parse_inferences(), _parse_pareja(), _parse_personas(), InferredAttribute, Análisis del CONTENIDO de cada foto (qué se ve: objetos, actividades,…, Comprobación barata (sin cargar el modelo) de si este módulo puede funcionar:… (+3 more)

### Community 32 - "Community 32"
Cohesion: 0.17
Nodes (5): mock, Tests del módulo de autenticación con Reddit. Las llamadas HTTP reales a Reddit…, Sin FRONTEND_ORIGIN configurada, la redirección final se deriva del Host de la…, test_callback_falls_back_to_request_host_without_frontend_origin(), test_reddit_callback_exchanges_token_and_sets_session()

### Community 33 - "Community 33"
Cohesion: 0.39
Nodes (6): cardStyle(), Landing(), PLATFORM_CARDS, PlatformCardData, relativeOffset(), Platform

### Community 34 - "Community 34"
Cohesion: 0.25
Nodes (10): embed_image(), load_model(), main(), _normalize_columns(), DataFrame, Extrae embeddings DINOv2 de todas las imágenes de España descargadas por…, El CSV real de OSV-5M puede nombrar las columnas de forma distinta a lo que…, Image (+2 more)

### Community 35 - "Community 35"
Cohesion: 0.35
Nodes (10): _dir_size_gb(), _download_and_extract_shard(), _load_all_ids(), main(), DataFrame, Lock, Path, Descarga TODAS las imágenes de OpenStreetView-5M (train + test), de CUALQUIER… (+2 more)

### Community 36 - "Community 36"
Cohesion: 0.18
Nodes (3): TestNationality, TestOccupation, TestUniversity

### Community 37 - "Community 37"
Cohesion: 0.18
Nodes (3): Asturias es una comunidad de una sola provincia: como su nombre ya coincide con…, Canarias' no es una provincia del INE (son dos: Las Palmas y Santa Cruz de…, TestLocation

### Community 38 - "Community 38"
Cohesion: 0.18
Nodes (10): dependencies, gh-pages, shx, name, scripts, build, deploy, version (+2 more)

### Community 39 - "Community 39"
Cohesion: 0.33
Nodes (4): final_remaining_population(), Nº de personas en España que compartirían, EN CONJUNTO, todos los rasgos…, Si un rasgo intermedio (p.ej. ubicación) no es estimable, el siguiente rasgo…, TestFinalRemainingPopulation

### Community 40 - "Community 40"
Cohesion: 0.39
Nodes (8): compute_score(), InferredAttribute, PrivacyScore, WritingFingerprint, Motor de scoring de exposición de privacidad. Pondera varias señales en una…, _score_deanonymization_ease(), _score_geolocation(), _score_inferable_data()

### Community 41 - "Community 41"
Cohesion: 0.22
Nodes (3): Sin comunidad autónoma conocida, la lengua materna cooficial no se puede acotar…, Si solo se conoce la provincia (no la comunidad autónoma directamente), debe…, TestLenguaMaternaStep

### Community 42 - "Community 42"
Cohesion: 0.20
Nodes (3): mockNavigate, ResizeObserverStub, AnalysisProgressEvent

### Community 43 - "Community 43"
Cohesion: 0.32
Nodes (7): _lifespan(), get, Punto de entrada de la aplicación. Diseño RGPD: no se conecta ninguna base de…, root(), _geolocation_available(), _lazy_load(), Comprobación barata (sin cargar el modelo ni el índice) de si el módulo de…

### Community 44 - "Community 44"
Cohesion: 0.38
Nodes (3): merge_findings(), Combina lo detectado por regex (determinista y gratuito) con lo detectado por…, TestMergeFindings

### Community 45 - "Community 45"
Cohesion: 0.38
Nodes (6): _load_metadata(), main(), DataFrame, Path, Reconstruye metadata.csv a partir de las imágenes YA DESCARGADAS y el CSV…, Misma lógica de caché que _load_spain_ids/_load_all_ids en los scripts de…

### Community 47 - "Community 47"
Cohesion: 0.29
Nodes (7): concurrently, @eslint/js, typescript, devDependencies, concurrently, @eslint/js, typescript

### Community 60 - "Community 60"
Cohesion: 0.50
Nodes (3): patch_spacy_model(), fixture, Fixtures compartidas de pytest. Nota: en este sandbox de desarrollo no siempre…

### Community 61 - "Community 61"
Cohesion: 0.67
Nodes (3): fixture, Cada test parte de una API key controlada explícitamente, en vez de depender…, reset_mistral_api_key()

### Community 64 - "Community 64"
Cohesion: 0.33
Nodes (5): vitest, AiSummaryUnavailableError, api, AiSummaryCard(), Status

## Knowledge Gaps
- **94 isolated node(s):** `name`, `version`, `build`, `deploy`, `gh-pages` (+89 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **37 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DemographicFindings` connect `Community 5` to `Community 0`, `Community 1`, `Community 2`, `Community 39`, `Community 8`, `Community 9`, `Community 41`, `Community 44`, `Community 13`, `Community 14`, `Community 46`, `Community 48`, `Community 22`, `Community 56`?**
  _High betweenness centrality (0.163) - this node is a cross-community bridge._
- **Why does `SocialPost` connect `Community 8` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 5`, `Community 7`, `Community 9`, `Community 10`, `Community 14`, `Community 17`, `Community 23`, `Community 26`, `Community 36`, `Community 37`, `Community 40`, `Community 44`, `Community 49`, `Community 51`, `Community 52`, `Community 53`, `Community 54`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Why does `InferredAttribute` connect `Community 0` to `Community 2`, `Community 3`, `Community 5`, `Community 8`, `Community 9`, `Community 40`, `Community 14`, `Community 15`, `Community 18`, `Community 50`, `Community 55`, `Community 24`, `Community 26`, `Community 29`, `Community 31`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Are the 26 inferred relationships involving `DemographicFindings` (e.g. with `AiExtractionUnavailable` and `InferredAttribute`) actually correct?**
  _`DemographicFindings` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 36 inferred relationships involving `SocialPost` (e.g. with `InstagramClient` and `AiExtractionUnavailable`) actually correct?**
  _`SocialPost` has 36 INFERRED edges - model-reasoned connections that need verification._
- **What connects `name`, `version`, `build` to the rest of the system?**
  _94 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.09218807848944835 - nodes in this community are weakly interconnected._