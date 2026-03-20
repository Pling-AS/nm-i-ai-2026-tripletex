# Use a slim Python 3.11 base image
FROM python:3.11-slim-bookworm

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configure environment for uv and Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Set the working directory
WORKDIR /app

# Install dependencies first (cached layer)
# This uses uv.lock for deterministic builds
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application code
COPY src ./src
COPY task_docs ./task_docs
COPY README.md ./

# Install the project itself (creates the virtual environment in .venv)
RUN uv sync --frozen --no-dev

# Place the virtual environment in the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
# We use shell form to allow variable expansion for $PORT
# Default PORT is 8000 if not set
CMD uvicorn tripletex_agent.main:app --host 0.0.0.0 --port ${PORT:-8000}
