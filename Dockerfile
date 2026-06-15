# Stage 1 - Builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Instalo dependencias de compilación
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    entr \
    && rm -rf /var/lib/apt/lists/*

# Configuración de Poetry para instalar en el Python global del sistema del Builder
ENV POETRY_VERSION=2.3.2 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1

# Instalar Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copiar archivos de dependencias
COPY pyproject.toml poetry.lock ./

# Instalar dependencias del proyecto en el entorno global del Builder
RUN poetry install --no-root --no-interaction --no-ansi

# Stage 2 - Runtime
FROM python:3.12-slim

LABEL maintainer="eldie1984@gmail.com" \
      description="ML Engineering Challenge - Taligent" \
      version="1.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instalo dependencias de runtime (necesarias para psycopg2 y tu entrypoint)
RUN apt-get update && apt-get install -y \
    libpq5 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# ¡CLAVE! Copiamos las librerías instaladas por Poetry desde el Python del Builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copio código fuente y estructura
COPY src/ /app/src/
COPY entrypoint.sh .

# Crear directorios para outputs y asegurar permisos
RUN mkdir -p  logs

# Crear usuario no-root (seguridad) y asignar permisos en un solo paso
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /home/appuser -m appuser && \
    chown -R appuser:appuser /app && \
    chmod +x entrypoint.sh

# Cambiar a usuario no-root
USER appuser

# Health check (ahora sí encontrará pandas si decide verificar algo más complejo)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys-import pandas; sys.exit(0)" || exit 1

# Entrypoint
ENTRYPOINT ["./entrypoint.sh"]