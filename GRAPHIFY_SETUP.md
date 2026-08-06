# Graphify: configuración y replicación en un PC nuevo

Este proyecto usa [Graphify](https://github.com/Graphify-Labs/graphify) para generar un grafo
de conocimiento del código (`graphify-out/`). Claude Code lo consulta automáticamente en vez de
releer archivos sueltos, lo que reduce el consumo de tokens en las sesiones de desarrollo.

## Qué se versiona y qué no

| Ruta | ¿Se versiona? | Motivo |
|---|---|---|
| `graphify-out/graph.json` | Sí | El grafo en sí. Portátil (rutas relativas). |
| `graphify-out/graph.html` | Sí | Visualización interactiva del grafo. |
| `graphify-out/GRAPH_REPORT.md` | Sí | Resumen legible (hubs, comunidades, conexiones). |
| `graphify-out/manifest.json` | Sí | Portátil desde v0.9+; evita una reconstrucción completa al clonar. |
| `graphify-out/cost.json` | **No** | Coste de tokens de la última ejecución, solo informativo local. |
| `graphify-out/cache/` | **No** | Caché de extracción (opcional, se regenera sola). |
| `.claude/skills/graphify/` | Sí | El skill en sí, no depende de la máquina. |
| `CLAUDE.md` (raíz) y `.claude/CLAUDE.md` | Sí | Instrucciones para Claude Code, sin rutas absolutas. |
| `.claude/settings.json` | **No** | ⚠️ Contiene la **ruta absoluta al ejecutable `graphify` de esa máquina**
(p. ej. `C:/Users/<usuario>/AppData/.../graphify.EXE` en Windows, `~/.local/bin/graphify` en Linux/Mac).
Si se versiona, deja de funcionar en cualquier otro equipo o usuario. Se regenera en 5 segundos (ver abajo). |

Todo esto ya está reflejado en `.gitignore`.

## Instalación en un PC nuevo (checklist)

1. **Clona el repo** con `graphify-out/` ya incluido — no hace falta reconstruir el grafo desde cero.

2. **Instala Graphify con `uv tool install` o `pipx`, nunca con `pip` a secas:**

   ```bash
   uv tool install graphifyy      # recomendado
   # o
   pipx install graphifyy
   ```

   `pip install graphifyy` funciona, pero el skill resuelve el intérprete de Python en
   tiempo de ejecución desde `graphify-out/.graphify_python`; si luego cambias de entorno
   virtual o reinstalas con otra herramienta, da `ModuleNotFoundError`. `uv tool`/`pipx`
   aíslan el paquete en su propio entorno y evitan el problema.

   Si `graphify` no se encuentra tras instalar:
   ```bash
   uv tool update-shell   # o: pipx ensurepath
   ```
   y abre una terminal nueva.

3. **Regenera `.claude/settings.json` para esta máquina** (obligatorio, es el paso que
   falla si simplemente clonas y ya):

   ```bash
   graphify claude install --project
   ```

   Esto sobrescribe `.claude/settings.json` con la ruta correcta del `graphify` recién
   instalado en *este* equipo. No hace falta tocar nada más: `.claude/CLAUDE.md`,
   `CLAUDE.md` y `.claude/skills/` ya están en el repo y no cambian entre máquinas.

4. **Verifica que el grafo está al día con el commit actual:**

   ```bash
   git rev-parse HEAD
   # compáralo con la línea "Built from commit" al principio de graphify-out/GRAPH_REPORT.md
   ```

   Si hay código nuevo desde esa fecha:

   ```bash
   graphify update .        # solo AST, sin coste de API
   ```

5. **(Opcional) Hook de git para que el grafo se reconstruya solo tras cada commit:**

   ```bash
   graphify hook install
   ```

   Este comando también embebe una ruta absoluta al intérprete, así que **hay que
   volver a ejecutarlo en cada máquina nueva** (y tras cada `upgrade` de graphify).

## Estructura correcta (para no repetir el lío de duplicados)

Todo cuelga de la raíz del repo, en un único sitio cada cosa — nunca anidado dentro de
una carpeta `graphify/` propia ni duplicado en dos rutas:

```
AmITraceable/
├── CLAUDE.md                      # instrucciones generales (versionado)
├── GRAPHIFY_SETUP.md              # este documento
├── .claude/
│   ├── CLAUDE.md                  # versionado
│   ├── settings.json              # NO versionado (por máquina)
│   └── skills/graphify/           # versionado
└── graphify-out/
    ├── graph.json                 # versionado
    ├── graph.html                 # versionado
    ├── GRAPH_REPORT.md            # versionado
    ├── manifest.json              # versionado
    ├── cost.json                  # NO versionado
    └── cache/                     # NO versionado
```

Si en algún momento vuelve a aparecer una carpeta `graphify/` con una copia anidada de
`graphify-out/` o `.claude/` dentro (típico de descomprimir un zip de resultados sin
fusionar carpetas), hay que borrarla y quedarse solo con la de la raíz.

## Ampliar el grafo con documentación (Arc42, `.adoc`, PDFs)

Por ahora el grafo solo indexa código (`--code-only`), que es gratis y local (tree-sitter,
sin LLM). Los capítulos Arc42 (`docs/src/*.adoc`) no están incluidos porque requieren una
pasada semántica con un modelo. Dentro de una sesión de Claude Code es gratis (usa el
modelo de la sesión); en modo headless (CI, terminal) hace falta una API key:

```bash
ANTHROPIC_API_KEY=sk-... graphify extract ./docs --backend claude
```

Antes de decidir si merece la pena, ten en cuenta el coste de tokens de esa pasada
semántica frente al beneficio (los `.adoc` ya son relativamente compactos y legibles
directamente).
