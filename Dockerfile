# Stage 1: Builder
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable

COPY src ./src

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY configs ./configs

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 9090

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import tessera; print(tessera.__version__)" || exit 1

ENTRYPOINT ["tessera"]
CMD ["--help"]
