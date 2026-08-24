FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra profiling --no-install-project

COPY profiling/dashboard_profiling.py /app
COPY config/ ./config/
RUN uv sync --frozen --no-dev --extra profiling

CMD ["streamlit", "run", "dashboard_profiling.py", "--server.port=8501", "--server.address=0.0.0.0"]