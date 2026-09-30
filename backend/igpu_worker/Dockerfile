# Imagen completamente aparte de backend/Dockerfile -- a propósito, ver el
# docstring de app.py: este proceso NUNCA debe compartir site-packages con
# el backend (torch-directml exige una versión de torch incompatible con
# el build CUDA que usa Moondream2). Solo se construye/arranca con
# `docker compose --profile igpu up` (ver docker-compose.yml).
#
# Migrado de pip a uv: dependencias desde pyproject.toml + uv.lock de este mismo
# directorio (generar con `uv lock`). SIN VERIFICAR: no se ha podido construir
# desde el entorno donde se escribió -- probar `docker compose --profile igpu build`.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# Sin UV_COMPILE_BYTECODE: fallo en una build real ("Python startup timed out
# (60s)" al precompilar torch con la maquina cargada) y alarga la build.
# UV_PROJECT_ENVIRONMENT fuera de /app; UV_HTTP_TIMEOUT sustituye al
# `--timeout 120` que se le pasaba a pip (descargas grandes de torch).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120 \
    PATH="/opt/venv/bin:${PATH}"
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

COPY app.py ./

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
