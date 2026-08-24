FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra gradio --no-install-project

COPY src/ ./src/
COPY config/ ./config/
COPY data/raw/application_test.csv ./data/raw/application_test.csv
RUN uv sync --frozen --no-dev --extra gradio

CMD ["python", "-m", "src.frontend.formulaire"]
