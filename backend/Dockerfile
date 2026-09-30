# Migrado de pip a uv (https://docs.astral.sh/uv/). SIN VERIFICAR TODAVIA: esta
# version del Dockerfile no se ha podido construir ni arrancar desde el entorno
# donde se escribio (sin Docker ni GPU) -- probar `docker compose build backend`
# antes de fiarse. Requiere `uv.lock` en backend/ (generar con `uv lock`).
#
# Que cambia respecto a la version con pip:
#   - Dependencias: `uv sync --locked` desde pyproject.toml/uv.lock (versiones
#     exactas, reproducibles) en vez de `pip install -r requirements*.txt`.
#     WITH_GEOLOCATION=true instala ademas el extra `vision`.
#   - Python 3.12 lo instala uv (`uv python install`), ya no hace falta el PPA
#     deadsnakes ni `software-properties-common`. Tampoco el workaround de
#     `--ignore-installed blinker`: venia de `python3-blinker` (apt, instalado
#     por software-properties-common); el venv de uv esta aislado del Python
#     del sistema.
#   - Modelos de spaCy: ya son dependencias del proyecto (pyproject.toml), no
#     hay `python -m spacy download` (necesitaria `pip` dentro del venv).
#   - Se mantiene la etapa `cuda-builder` con la compilacion de llama-cpp-python
#     con CUDA para Turing (cc 7.5). Se elimina la de `llama-quantize` (codigo
#     muerto, ver nota en la etapa 1): recorta varios minutos del build en frio.
#
# Historia (sigue vigente): el wheel PRECOMPILADO de `llama-cpp-python` (indice
# cu130 de abetlen) cargaba el modelo pero moria con SIGILL (exit 132) en
# cuanto tocaba GPU en la GTX 1650 real; en CPU pura si cargaba, asi que el
# problema era ese wheel con CUDA (probablemente sin codigo para compute
# capability 7.5). Se compila desde codigo fuente pidiendo esa arquitectura.

# ---- Etapa 1: compilar llama-cpp-python con soporte CUDA para Turing ----
# Imagen "devel" (trae nvcc y las cabeceras de CUDA) SOLO para esta etapa --
# no llega a la imagen final, asi que no infla el tamano de la imagen que de
# verdad se despliega (ver etapa 2, "runtime", mucho mas ligera).
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS cuda-builder

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# DEBIAN_FRONTEND=noninteractive: sin esto, `tzdata` pregunta la zona horaria
# de forma interactiva y la build se queda colgada para siempre (confirmado en
# un build real, 12/9). `TZ=Etc/UTC` de paso, para no dejarlo en un valor por
# defecto no elegido.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON=3.12 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && uv python install 3.12

# 75 = compute capability de la GTX 1650 (Turing), confirmada en el log real de
# `ggml_cuda_init` ("compute capability 7.5"). El wheel se construye con
# `--no-deps` de todos modos: solo nos interesa ESTE wheel compilado a medida
# -- el resto de dependencias de llama-cpp-python (numpy, diskcache,
# typing-extensions, jinja2 -- todas puro Python) se instalan normal en la
# imagen final, sin compilar nada.
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75"

# Descarga el sdist a mano y lo extrae para construir el wheel con `uv build`.
# uv no tiene equivalente de `pip download`, asi que ese unico paso usa pip de
# forma efimera (`uvx --from pip`, no queda instalado en ningun sitio); el
# wheel se construye con `uv build`.
RUN uvx --from pip pip download --no-binary :all: --no-deps --no-cache-dir \
      -d /src "llama-cpp-python>=0.3,<0.4" \
    && mkdir -p /src/extracted \
    && tar --strip-components=1 -xzf /src/llama_cpp_python-*.tar.gz -C /src/extracted \
    && uv build --wheel --python 3.12 --out-dir /wheels /src/extracted

# NOTA (build mas rapido): aqui se compilaba tambien `llama-quantize` (un SEGUNDO
# build completo de ggml-cuda para cc 7.5, varios minutos) para cuantizar
# Moondream2 en el servidor. Ya no lo usa nada: ai_client.py y scene_analysis.py
# descargan GGUF YA cuantizados de Hugging Face (Qwen3.5-4B Q4_K_M, ver ADR-50),
# `_ensure_quantized_model()` ya no existe y ninguna ruta de app/ llama al
# binario. Eliminado junto con su copia a la imagen final y los ENV
# PATH/LD_LIBRARY_PATH que apuntaban a /opt/llama-quantize-bin. Si algun dia
# hiciera falta cuantizar en el servidor, recuperarlo del historial de git.


# ---- Etapa 2: imagen final ----
# Imagen "runtime" de NVIDIA (Ubuntu 22.04) -- SIN el toolkit completo de
# compilacion (mucho mas ligera que la de la etapa 1), pero CON las librerias
# compartidas de CUDA (libcudart y companias) que el wheel compilado en la
# etapa 1 necesita en tiempo de ejecucion. `torch` no tenia este problema
# porque sus wheels llevan el runtime de CUDA empaquetado --
# `llama-cpp-python` compilado desde fuente NO.
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# Mismo motivo y mismo fallo real que en la etapa 1 para DEBIAN_FRONTEND/TZ (es
# otra imagen base con su propio apt-get, no hereda el ENV de la otra etapa).
# UV_PROJECT_ENVIRONMENT: el venv vive fuera de /app (que se solapa con el
# volumen ./backend/data:/app/data en docker-compose.yml).
# Sin UV_COMPILE_BYTECODE a proposito: precompilar los .pyc de torch/CUDA tarda
# mucho y fallo en una build real con "Python startup timed out (60s)" cuando la
# maquina iba cargada (Docker Desktop/WSL2 extrayendo capas en paralelo). Sin el,
# Python compila en el primer import; coste despreciable.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON=3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120

# build-essential: por si alguna dependencia sin wheel hay que compilarla.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && uv python install 3.12

# Geolocalizacion por imagen (opcional, ver app/vision/geolocation.py): el
# extra `vision` (torch/faiss/transformers/ctranslate2...) pesa varios cientos
# de MB y solo aporta algo si ademas tienes el indice FAISS ya construido (ver
# scripts/geolocalization/build_faiss_index.py). Por defecto NO se instala,
# para que la imagen siga ligera si solo usas Reddit o Instagram sin
# geolocalizacion. Para activarlo:
#   docker compose build --build-arg WITH_GEOLOCATION=true backend
#
# OJO: WITH_GEOLOCATION=false YA NO evita el coste de partir de una imagen base
# de NVIDIA -- la imagen base es `nvidia/cuda` incondicionalmente. Se acepta
# este trade-off por simplicidad; si algun dia interesa una imagen ligera de
# verdad sin geolocalizacion, se puede separar en dos etapas finales
# seleccionables con `--target`.
ARG WITH_GEOLOCATION=false

# Solo pyproject.toml + uv.lock antes que el codigo: la capa de dependencias
# (la mas lenta) se reutiliza mientras no cambien. `--locked` falla si
# uv.lock no esta al dia con pyproject.toml (en vez de resolver en silencio).
# `--no-dev`: sin pytest & co. en la imagen. La cache de uv se monta como
# cache de BuildKit (no queda en la imagen).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_GEOLOCATION" = "true" ]; then EXTRAS="--extra vision"; else EXTRAS=""; fi; \
    uv sync --locked --no-dev $EXTRAS

# llama-cpp-python (wheel compilado a medida en la etapa 1). Se instala DESPUES de
# `uv sync` y con `uv pip install` a proposito: no esta en uv.lock (el wheel de
# PyPI/abetlen no sirve para Turing, ver arriba). NO ejecutar `uv sync` (ni
# `uv run`) despues dentro de la imagen: eliminaria este paquete por no estar en
# el lock. El CMD usa el `python` del venv directamente.
COPY --from=cuda-builder /wheels /wheels
RUN if [ "$WITH_GEOLOCATION" = "true" ]; then \
      uv pip install --python /opt/venv/bin/python /wheels/*.whl; \
    fi
ENV PATH="/opt/venv/bin:${PATH}"

COPY app ./app

EXPOSE 3000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
