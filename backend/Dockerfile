# SIN VERIFICAR TODAVÍA (12/9, ver conversación con Claude) -- no tengo
# GPU ni acceso a los registros de NVIDIA en el entorno donde se escribió
# esto, así que este Dockerfile no se ha podido construir ni arrancar de
# verdad antes de dártelo. Motivo del cambio: el wheel PRECOMPILADO de
# `llama-cpp-python` (índice cu130 de abetlen, ver requirements-vision.txt
# de antes) cargaba el modelo pero moría con SIGILL (exit 132) en cuanto
# tocaba GPU -- aislado con una prueba en CPU pura, que SÍ cargó bien, así
# que el problema es específico de ESE wheel con CUDA, no de tu CPU ni de
# la GPU en sí. La causa más probable: el wheel no trae código compilado
# para compute capability 7.5 (Turing, tu GTX 1650) entre sus arquitecturas
# soportadas. Aquí se compila desde código fuente, pidiendo explícitamente
# esa arquitectura -- no hay forma de estar seguro de que esto arregla el
# problema sin probarlo en tu máquina real.

# ---- Etapa 1: compilar llama-cpp-python con soporte CUDA para Turing ----
# Imagen "devel" (trae nvcc y las cabeceras de CUDA) SOLO para esta etapa
# -- no llega a la imagen final, así que no infla el tamaño de la imagen
# que de verdad se despliega (ver etapa 2, "runtime", mucho más ligera).
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS cuda-builder

# DEBIAN_FRONTEND=noninteractive: SIN esto, `tzdata` (dependencia
# indirecta de `software-properties-common`/deadsnakes en esta imagen
# base, que no la trae ya configurada como sí hacía `python:3.12-slim`)
# pregunta la zona horaria de forma interactiva y la build se queda
# colgada para siempre esperando un input que nunca llega -- confirmado
# en un build real (12/9), no es hipotético. `TZ=Etc/UTC` de paso, para
# no dejarlo en un valor por defecto no elegido.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Python 3.12 vía deadsnakes: Ubuntu 22.04 trae 3.10 de serie, y el resto
# del proyecto (imagen final, requirements.txt) está pensado para 3.12
# (antes venía gratis de partir de `python:3.12-slim`) -- aquí hay que
# instalarlo a mano para que el wheel compilado tenga la ABI (cp312)
# correcta y sea instalable en la imagen final.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m ensurepip --upgrade

# 75 = compute capability de la GTX 1650 (Turing), confirmada en el log
# real de `ggml_cuda_init` ("compute capability 7.5"). `--no-deps`: solo
# nos interesa ESTE wheel compilado a medida -- el resto de dependencias
# de llama-cpp-python (numpy, diskcache, typing-extensions -- todas puro
# Python) se instalan normal en la imagen final, sin necesitar compilar
# nada.
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75"

# Descarga el sdist a mano (en vez de `pip wheel` directo) y lo extrae a
# una ruta que sobrevive después del build del wheel -- así el código
# fuente de `llama.cpp` que trae vendorizado (submódulo
# `vendor/llama.cpp`, MISMA versión exacta que usa el wheel de Python)
# sigue disponible para compilar también `llama-quantize` con él más
# abajo. Si se hiciera con `pip wheel` a secas, ese código fuente se
# limpiaría solo al terminar y no habría forma de reutilizarlo para
# construir la herramienta CLI de cuantización con garantía de que hablan
# el mismo formato GGUF exacto que el wheel ya instalado.
RUN python3.12 -m pip download --no-binary :all: --no-deps --no-cache-dir \
      -d /src "llama-cpp-python>=0.3,<0.4" \
    && mkdir -p /src/extracted \
    && tar --strip-components=1 -xzf /src/llama_cpp_python-*.tar.gz -C /src/extracted \
    && python3.12 -m pip wheel --no-cache-dir --no-deps -w /wheels /src/extracted

# `llama-quantize`: para cuantizar el GGUF F16 a Q8_0 (ver historial en
# scene_analysis.py, junto a `_GGUF_REPO_ID`, sobre por qué Q8_0 y no un
# quant ya hecho de un checkpoint más antiguo).
#
# CONFIRMADO EN PRODUCCIÓN (15/9, sesión de depuración manual dentro de
# la propia etapa `cuda-builder`, ver conversación con Claude): los dos
# supuestos originales (ruta `vendor/llama.cpp` dentro del sdist, nombre
# de la opción `LLAMA_BUILD_TOOLS`) eran correctos -- `cmake` configuraba
# bien. El fallo real era de ENLAZADO: `libggml-cuda.so` necesita
# `libcuda.so` (la librería del DRIVER de NVIDIA, no del toolkit), que no
# existe durante `docker build` (sin GPU real conectada en ese momento,
# solo la hay en tiempo de `docker run`). El toolkit de CUDA trae un
# "stub" de esa librería pensado exactamente para esto
# (`/usr/local/cuda/lib64/stubs/libcuda.so`) -- el wheel de Python lo
# encuentra solo (vía `FindCUDAToolkit` de CMake moderno), pero el target
# `llama-quantize` no, así que hay que indicárselo a mano con
# `CMAKE_EXE_LINKER_FLAGS`. Confirmado que compila limpio con esto.
#
# BEST-EFFORT DE TODAS FORMAS (el `|| echo ... ; true` al final, en su
# propio RUN separado del wheel de arriba): aunque ya verificado en este
# entorno concreto, se deja como best-effort por si una versión futura de
# llama-cpp-python cambia el layout o el nombre de la opción -- si falla,
# NO debe tirar abajo toda la build. El wheel de `llama-cpp-python` (lo
# único imprescindible para que la app funcione en absoluto, aunque sea
# sin cuantizar) ya se construyó en el paso anterior y no depende de
# este. `scene_analysis.py` comprueba si el binario existe antes de
# intentar usarlo y cae a F16 sin más si no está (ver
# `_ensure_quantized_model()`).
# CONFIRMADO EN PRODUCCIÓN (16/9): `llama-quantize` no es un ejecutable
# suelto -- depende de varias `.so` hermanas (`libllama-quantize-impl.so`
# y compañía) que quedan en la MISMA carpeta `build/bin/` al compilar.
# Copiar solo el binario (como se hacía antes) lo deja sin esas
# dependencias -- "error while loading shared libraries" en tiempo de
# ejecución, no al copiar ni al arrancar el contenedor. Se copia la
# carpeta `bin/` ENTERA (no solo el ejecutable) para que seguidas
# conserven la resolución de rutas relativas (RPATH `$ORIGIN`) que ya
# trae el build por defecto.
RUN cmake -S /src/extracted/vendor/llama.cpp -B /src/extracted/vendor/llama.cpp/build \
        -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_BUILD_TOOLS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build /src/extracted/vendor/llama.cpp/build --target llama-quantize -j"$(nproc)" \
    && cp -r /src/extracted/vendor/llama.cpp/build/bin /wheels/llama-quantize-bin \
    || echo "AVISO: no se pudo compilar llama-quantize -- Moondream2 seguirá funcionando en F16, sin cuantizar. Revisar este paso del build manualmente." ; true


# ---- Etapa 2: imagen final ----
# Antes: `python:3.12-slim` (Debian). Ahora: imagen "runtime" de NVIDIA
# (Ubuntu 22.04) -- SIN el toolkit completo de compilación (mucho más
# ligera que la de la etapa 1), pero CON las librerías compartidas de
# CUDA (libcudart y compañía) que el wheel compilado en la etapa 1
# necesita en tiempo de ejecución, no solo para compilar. `torch` no
# tenía este problema porque sus wheels llevan el runtime de CUDA
# empaquetado dentro (ver conversación) -- `llama-cpp-python` compilado
# desde fuente NO, depende de que el sistema ya lo tenga.
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04
WORKDIR /app

# Mismo motivo y mismo fallo real que en la etapa 1 (ver ese comentario)
# -- se repite aquí porque es una imagen base distinta con su propio
# apt-get, no hereda el ENV de la otra etapa.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Mismo motivo que en la etapa 1: Python 3.12 no viene de serie en esta
# imagen base.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m ensurepip --upgrade \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Geolocalización por imagen (opcional, ver app/vision/geolocation.py):
# torch/faiss/transformers pesan varios cientos de MB y solo aportan algo
# si además tienes el índice FAISS ya construido (ver
# scripts/geolocalization/build_faiss_index.py). Por defecto NO se instalan, para que la
# imagen siga ligera si solo usas Reddit o Instagram sin geolocalización.
# Para activarlo:
#   docker compose build --build-arg WITH_GEOLOCATION=true backend
#
# OJO: con este cambio, WITH_GEOLOCATION=false YA NO evita el coste de
# partir de una imagen base de NVIDIA (antes sí evitaba por completo tocar
# CUDA) -- ahora la imagen base ya es `nvidia/cuda` incondicionalmente,
# aunque no se instale nada de `requirements-vision.txt`. Se acepta este
# trade-off por simplicidad (evitar un `Dockerfile` con selección de
# imagen base condicional, más complejo de mantener) -- si algún día
# vuelve a interesar una imagen ligera de verdad para el caso sin
# geolocalización, esto se puede separar en dos etapas finales
# seleccionables con `--target`.
ARG WITH_GEOLOCATION=false
COPY requirements-vision.txt ./
COPY --from=cuda-builder /wheels /wheels
RUN if [ "$WITH_GEOLOCATION" = "true" ]; then \
      python -m pip install --no-cache-dir /wheels/*.whl && \
      python -m pip install --no-cache-dir --timeout 120 --retries 5 -r requirements-vision.txt && \
      if [ -d /wheels/llama-quantize-bin ]; then \
        cp -r /wheels/llama-quantize-bin /opt/llama-quantize-bin && \
        chmod +x /opt/llama-quantize-bin/llama-quantize; \
      else \
        echo "AVISO: llama-quantize no se compiló en la etapa anterior -- Moondream2 correrá en F16, sin cuantizar (ver scene_analysis.py, _ensure_quantized_model())"; \
      fi; \
    fi
# Añadido incondicionalmente (no dentro del `if` de arriba): si
# WITH_GEOLOCATION=false, /opt/llama-quantize-bin no existe, y
# `PATH` con un directorio inexistente no da ningún error, `which
# llama-quantize` simplemente no lo encuentra -- mismo comportamiento
# best-effort que ya tiene `scene_analysis.py`.
ENV PATH="/opt/llama-quantize-bin:${PATH}"
# CONFIRMADO EN PRODUCCIÓN (16/9): copiar el binario junto a sus `.so`
# hermanas en la MISMA carpeta no bastó -- seguía fallando con "cannot
# open shared object file". El binario lleva grabada de fábrica una ruta
# RPATH que parece ABSOLUTA al directorio de compilación original
# (`/src/extracted/...`, que no existe en la imagen final), no relativa
# a sí mismo (`$ORIGIN`) -- así que estar "al lado" no es suficiente,
# hay que decírselo explícitamente al enlazador dinámico en tiempo de
# ejecución. `LD_LIBRARY_PATH` no depende de lo que lleve grabado el
# binario: el enlazador sigue probando rutas de búsqueda aunque el RPATH
# grabado apunte a un sitio que ya no existe.
ENV LD_LIBRARY_PATH="/opt/llama-quantize-bin:${LD_LIBRARY_PATH}"

# Modelos de spaCy necesarios para el fingerprinting de escritura (ES/EN)
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download es_core_news_sm

COPY app ./app

EXPOSE 3000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
