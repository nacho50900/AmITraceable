# Imagen completamente aparte de backend/Dockerfile -- a propósito, ver el
# docstring de app.py: este proceso NUNCA debe compartir site-packages con
# el backend (torch-directml exige una versión de torch incompatible con
# el build CUDA que usa Moondream2). Solo se construye/arranca con
# `docker compose --profile igpu up` (ver docker-compose.yml).
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

COPY app.py ./

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
