import os
import sys
import random
from dotenv import load_dotenv
import psycopg2
from clickhouse_driver import Client

# Шаг 1. Загружаем переменные окружения, поднимаясь на уровень выше к корню проекта
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))

# Читаем параметры системы
NUM_USERS = int(os.getenv("MODEL_NUM_USERS", 1000))
NUM_MOVIES = int(os.getenv("MODEL_NUM_ITEMS", 500))
EMBEDDING_DIM = int(os.getenv("MODEL_EMBEDDING_DIM", 16))


# --- ПОДКЛЮЧЕНИЕ К БАЗАМ ---
def get_postgres_conn():
    # Читаем строго из ОС. Если скрипт на ПК — там будет 127.0.0.1 (из .env через load_dotenv),
    # Если внутри Docker — Docker Compose подменит это значение на имя контейнера.
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB")
    )


def get_clickhouse_client():
    return Client(
        host=os.getenv("CLICKHOUSE_HOST"),
        port=int(os.getenv("CLICKHOUSE_PORT_NATIVE", 9000)),
        user=os.getenv("CLICKHOUSE_USER"),
        password=os.getenv("CLICKHOUSE_PASSWORD"),
        database=os.getenv("CLICKHOUSE_DB")
    )


def init_databases():
    print("Инициализация таблиц в PostgreSQL...")
    pg_conn = get_postgres_conn()
    with pg_conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
                    DROP TABLE IF EXISTS users CASCADE;
                    CREATE TABLE users
                    (
                        user_id     INT PRIMARY KEY,
                        age         INT     NOT NULL,
                        has_premium BOOLEAN NOT NULL,
                        embedding   vector(16)
                    );
                    """)
        cur.execute(f"""
            DROP TABLE IF EXISTS movies CASCADE;
            CREATE TABLE movies (
                movie_id INT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                age_rating INT NOT NULL,
                genre_type INT NOT NULL,
                embedding vector({EMBEDDING_DIM})
            );
        """)
    pg_conn.commit()
    pg_conn.close()

    print("Инициализация базы данных в ClickHouse...")
    # Системный клиент тоже берет хост строго из операционной системы!
    ch_sys_client = Client(
        host=os.getenv("CLICKHOUSE_HOST"),
        port=int(os.getenv("CLICKHOUSE_PORT_NATIVE", 9000)),
        user=os.getenv("CLICKHOUSE_USER"),
        password=os.getenv("CLICKHOUSE_PASSWORD")
    )
    ch_sys_client.execute(f"CREATE DATABASE IF NOT EXISTS {os.getenv('CLICKHOUSE_DB')}")

    ch_client = get_clickhouse_client()
    ch_client.execute("DROP TABLE IF EXISTS watch_logs")
    ch_client.execute("""
                      CREATE TABLE watch_logs
                      (
                          user_id                UInt32,
                          movie_id               UInt32,
                          watched_seconds        UInt32,
                          movie_duration_seconds UInt32,
                          timestamp              DateTime DEFAULT now()
                      ) ENGINE = MergeTree()
        ORDER BY (user_id, timestamp)
                      """)


# --- ГЕНЕРАЦИЯ СИНТЕТИКИ ---
def generate_and_insert_data():
    pg_conn = get_postgres_conn()
    ch_client = get_clickhouse_client()

    print(f"Генерация {NUM_USERS} пользователей...")
    users_data = []
    user_age_map = {}  # Память для генерации логов

    for uid in range(NUM_USERS):
        # Делим юзеров на 3 возрастные группы
        rand_type = random.random()
        if rand_type < 0.2:  # 20% дети
            age = random.randint(6, 13)
        elif rand_type < 0.5:  # 30% подростки
            age = random.randint(14, 17)
        else:  # 50% взрослые
            age = random.randint(18, 65)

        has_premium = random.choice([True, False])
        users_data.append((uid, age, has_premium))
        user_age_map[uid] = age

    with pg_conn.cursor() as cur:
        cur.executemany("INSERT INTO users (user_id, age, has_premium) VALUES (%s, %s, %s)", users_data)

    print(f"Генерация {NUM_MOVIES} фильмов...")
    movies_data = []
    movie_meta_map = {}  # Память для генерации логов

    for mid in range(NUM_MOVIES):
        # Связываем жанр и возрастной ценз
        rand_genre = random.choice([0, 1, 2])  # 0-детский, 1-подростковый, 2-взрослый
        if rand_genre == 0:
            rating = random.choice([0, 6])
            title = f"Мультфильм {mid}"
        elif rand_genre == 1:
            rating = 12
            title = f"Экшен/Аниме {mid}"
        else:
            rating = random.choice([16, 18])
            title = f"Триллер/Драма {mid}"

        # Эмбеддинги при ините оставляем пустыми (NULL), их заполнит сервис обучения
        movies_data.append((mid, title, rating, rand_genre))
        movie_meta_map[mid] = {"rating": rating, "genre": rand_genre}

    with pg_conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO movies (movie_id, title, age_rating, genre_type, embedding) VALUES (%s, %s, %s, %s, NULL)",
            movies_data)

    pg_conn.commit()

    print("Генерация 50 000 аналитических логов просмотров в ClickHouse...")
    logs_data = []

    for _ in range(50000):
        uid = random.choice(list(user_age_map.keys()))
        mid = random.choice(list(movie_meta_map.keys()))

        user_age = user_age_map[uid]
        movie_genre = movie_meta_map[mid]["genre"]
        movie_rating = movie_meta_map[mid]["rating"]

        movie_len = random.randint(3600, 7200)  # Длина фильма от 1 до 2 часов

        # Симулируем паттерны поведения (Вкусы)
        # 1. Сценарий совпадения интересов
        if (user_age < 14 and movie_genre == 0) or \
                (14 <= user_age < 18 and movie_genre == 1) or \
                (user_age >= 18 and movie_genre == 2):
            # Посмотрел почти целиком (от 70% до 100% времени)
            watched = int(movie_len * random.uniform(0.7, 1.0))

        # 2. Сценарий "Не угадали с рекомендацией" (Жанр не тот)
        else:
            # Выключил в первые 15 минут (от 1% до 15%)
            watched = int(movie_len * random.uniform(0.01, 0.15))

        # 3. Аномалия: Ребенок зашел на фильм 18+ (Нарушение правил)
        if user_age < movie_rating:
            # В реальной жизни он либо успел посмотреть пару секунд и сработал блок,
            # либо обошел систему, но быстро выключил
            watched = random.randint(5, 60)  # от 5 до 60 секунд максимум

        logs_data.append((uid, mid, watched, movie_len))

    # Множественная быстрая вставка в ClickHouse
    ch_client.execute(
        "INSERT INTO watch_logs (user_id, movie_id, watched_seconds, movie_duration_seconds) VALUES",
        logs_data
    )

    pg_conn.close()
    print("Базы данных успешно наполнены синтетическими сценариями!")


if __name__ == "__main__":
    init_databases()
    generate_and_insert_data()
