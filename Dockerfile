FROM python:3.11-slim

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && apt-get install -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml poetry.lock ./

RUN poetry install --without dev --no-root

COPY .git/ ./.git/
COPY src/ ./src/
COPY params.yaml ./
COPY dvc.yaml ./
COPY .dvc/ ./.dvc/

CMD ["poetry", "run", "dvc", "repro"]