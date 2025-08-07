# --- Base image ---
FROM python:3.11-slim AS base

# --- System deps (мінімальний набір для faiss) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Workdir & requirements ---
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Копіюємо сам код ---
COPY ./src ./src

# --- Запуск FastAPI через Uvicorn ---
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
