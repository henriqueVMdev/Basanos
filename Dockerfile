FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Cache local do pip (BuildKit): rebuilds não re-baixam pacotes
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY *.py ./
COPY api ./api
COPY providers ./providers
COPY engine ./engine
COPY strategies ./strategies
COPY charts ./charts
COPY automation ./automation
COPY monte_carlo ./monte_carlo
COPY costs ./costs

EXPOSE 5000

CMD ["python", "server.py"]
