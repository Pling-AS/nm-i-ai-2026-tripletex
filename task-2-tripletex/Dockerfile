FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY task_docs ./task_docs

RUN pip install --upgrade pip && pip install .

CMD ["uvicorn", "tripletex_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
