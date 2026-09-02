FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml params.yaml ./
COPY src ./src
COPY frontend ./frontend
COPY scripts ./scripts

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/src
ENV SMART_GRID_API_URL=http://api:8000

EXPOSE 8000 8501

CMD ["python", "scripts/run_api.py"]
