FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

COPY backend/pyproject.toml backend/uv.lock /app/backend/
RUN uv sync --frozen --no-dev --no-install-project

COPY backend /app/backend
RUN uv sync --frozen --no-dev

EXPOSE 5001

CMD ["uv", "run", "gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]
