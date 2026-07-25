FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* ./
RUN pip install poetry --quiet && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-root --quiet

COPY src/ ./src/
COPY configs/ ./configs/

# Checkpoint copied in by deploy workflow
COPY checkpoints/ ./checkpoints/

ENV MODEL_PATH=checkpoints/best_model.pt
ENV RAIN_ALERT_THRESHOLD=0.5
ENV PORT=8000

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
