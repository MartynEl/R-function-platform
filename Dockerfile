FROM python:3.10-slim

# Устанавливаем системные утилиты
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    cron \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Увеличиваем тайм-аут до 1000 секунд для скачивания тяжелого PyTorch
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Копируем исходный код
COPY . .

CMD ["python"]

