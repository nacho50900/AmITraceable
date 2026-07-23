FROM python:3.12-slim
WORKDIR /app

# Dependencias de sistema necesarias para compilar scikit-learn/spaCy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Modelos de spaCy necesarios para el fingerprinting de escritura (ES/EN)
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download es_core_news_sm

COPY app ./app

EXPOSE 3000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
