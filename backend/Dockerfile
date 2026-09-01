FROM python:3.12-slim
WORKDIR /app

# Dependencias de sistema necesarias para compilar scikit-learn/spaCy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Geolocalización por imagen (opcional, ver app/vision/geolocation.py):
# torch/faiss/transformers pesan varios cientos de MB y solo aportan algo
# si además tienes el índice FAISS ya construido (ver
# scripts/geolocalization/build_faiss_index.py). Por defecto NO se instalan, para que la
# imagen siga ligera si solo usas Reddit o Instagram sin geolocalización.
# Para activarlo:
#   docker compose build --build-arg WITH_GEOLOCATION=true backend
ARG WITH_GEOLOCATION=false
COPY requirements-vision.txt ./
RUN if [ "$WITH_GEOLOCATION" = "true" ]; then \
      pip install --no-cache-dir --timeout 120 --retries 5 -r requirements-vision.txt; \
    fi

# Modelos de spaCy necesarios para el fingerprinting de escritura (ES/EN)
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download es_core_news_sm

COPY app ./app

EXPOSE 3000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
