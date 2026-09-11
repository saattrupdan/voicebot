FROM python:3.14-slim-bookworm

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        alsa-utils git libportaudio2 mpg123 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/

WORKDIR /project

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "--frozen", "--no-dev", "src/scripts/run_bot.py"]
