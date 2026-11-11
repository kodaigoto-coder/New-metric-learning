FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
RUN pip install --upgrade pip && pip install .

COPY app /app/app

EXPOSE 18501

CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=18501", "--server.headless=true", "--server.maxUploadSize=2000"]
