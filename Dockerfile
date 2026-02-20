FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

# Model is downloaded at startup via MODEL_URL env var
CMD ["sh", "-c", "uv run python download_model.py && uv run uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]

EXPOSE 7860
