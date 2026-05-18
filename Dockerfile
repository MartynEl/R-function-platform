FROM python:3.10-slim

# Устанавливаем системные утилиты для сборки С-библиотек (нужно для psycopg2 и clickhouse)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    cron \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем и устанавливаем общие зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь исходный код проекта в контейнер
COPY . .

# По умолчанию контейнер ничего не делает, команду мы переопределим в docker-compose
CMD ["python"]
