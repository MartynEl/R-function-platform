import os
import sys
import random
import numpy as np
from dotenv import load_dotenv
from clickhouse_driver import Client
from pymilvus import MilvusClient, DataType

# Шаг 1. Загружаем переменные окружения, поднимаясь на уровень выше к корню проекта
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))

# Читаем параметры системы
NUM_USERS = int(os.getenv("MODEL_NUM_USERS", 1000))
NUM_MOVIES = int(os.getenv("MODEL_NUM_ITEMS", 500))
EMBEDDING_DIM = int(os.getenv("MODEL_EMBEDDING_DIM", 16))
MILVUS_DB_PATH = "./milvus_pro_demo.db"


# --- ПОДКЛЮЧЕНИЕ К БАЗАМ ---
def get_clickhouse_client():
    return Client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKHOUSE_PORT_NATIVE", 9000)),
        user=os.getenv("CLICKHOUSE_USER", "martin"),
        password=os.getenv("CLICKHOUSE_PASSWORD", "clickhouse_secure_pass_789"),
        database=os.getenv("CLICKHOUSE_DB", "r_analytics_db")
    )


def init_databases():
    print("Инициализация схемы в Milvus Lite...")
    milvus_client = MilvusClient(MILVUS_DB_PATH)

    # Свежий перезапуск коллекций для чистых тестов
    if milvus_client.has_collection("users"):
        milvus_client.drop_collection("users")
    if milvus_client.has_collection("movies"):
        milvus_client.drop_collection("movies")

    # 1. Схема и создание коллекции пользователей
    schema_u = MilvusClient.create_schema(auto_id=False)
    schema_u.add_field(field_name="user_id", datatype=DataType.INT64, is_primary=True)
    schema_u.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    milvus_client.create_collection(collection_name="users", schema=schema_u)

    # 2. Схема и создание коллекции фильмов с метаданными и косинусным индексом
    schema_m = MilvusClient.create_schema(auto_id=False)
    schema_m.add_field(field_name="movie_id", datatype=DataType.INT64, is_primary=True)
    schema_m.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=255)
    schema_m.add_field(field_name="age_rating", datatype=DataType.INT64)
    schema_m.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name="embedding", metric_type="COSINE", index_type="FLAT")

    milvus_client.create_collection(
        collection_name="movies",
        schema=schema_m,
        index_params=index_params
    )

    print("Инициализация структуры таблиц в ClickHouse...")
    ch_sys_client = Client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKHOUSE_PORT_NATIVE", 9000)),
        user=os.getenv("CLICKHOUSE_USER", "martin"),
        password=os.getenv("CLICKHOUSE_PASSWORD", "clickhouse_secure_pass_789")
    )
    ch_sys_client.execute(f"CREATE DATABASE IF NOT EXISTS {os.getenv('CLICKHOUSE_DB', 'r_analytics_db')}")

    ch_client = get_clickhouse_client()
    ch_client.execute("DROP TABLE IF EXISTS watch_logs")
    ch_client.execute("DROP TABLE IF EXISTS users_metadata")
    ch_client.execute("DROP TABLE IF EXISTS movies_metadata")

    # Логи
    ch_client.execute("""
                      CREATE TABLE watch_logs
                      (
                          user_id                UInt32,
                          movie_id               UInt32,
                          watched_seconds        UInt32,
                          movie_duration_seconds UInt32,
                          timestamp              DateTime DEFAULT now()
                      ) ENGINE = MergeTree() ORDER BY (user_id, timestamp)
                      """)

    # Метаданные пользователей для dbt-трансформаций
    ch_client.execute("""
                      CREATE TABLE users_metadata
                      (
                          user_id     UInt32,
                          age         UInt8,
                          has_premium UInt8
                      ) ENGINE = MergeTree() ORDER BY user_id
                      """)

    # Метаданные фильмов для dbt-трансформаций
    ch_client.execute("""
                      CREATE TABLE movies_metadata
                      (
                          movie_id   UInt32,
                          title      String,
                          age_rating UInt8,
                          genre_type UInt8
                      ) ENGINE = MergeTree() ORDER BY movie_id
                      """)


def generate_and_insert_data():
    ch_client = get_clickhouse_client()
    milvus_client = MilvusClient(MILVUS_DB_PATH)

    print(f"Генерация {NUM_USERS} пользователей...")
    ch_users_data = []
    milvus_users_data = []
    user_age_map = {}

    for uid in range(NUM_USERS):
        rand_type = random.random()
        if rand_type < 0.2:
            age = random.randint(6, 13)
        elif rand_type < 0.5:
            age = random.randint(14, 17)
        else:
            age = random.randint(18, 65)
        has_premium = 1 if random.choice([True, False]) else 0

        ch_users_data.append((uid, age, has_premium))
        user_age_map[uid] = age

        # Генерируем случайный начальный вектор (единичный радиус для корректной работы COSINE)
        vec = np.random.normal(0, 0.1, EMBEDDING_DIM)
        vec = (vec / np.linalg.norm(vec)).tolist()
        milvus_users_data.append({"user_id": uid, "embedding": vec})

    ch_client.execute("INSERT INTO users_metadata (user_id, age, has_premium) VALUES", ch_users_data)
    milvus_client.insert(collection_name="users", data=milvus_users_data)

    print(f"Генерация {NUM_MOVIES} фильмов...")
    ch_movies_data = []
    milvus_movies_data = []
    movie_meta_map = {}

    for mid in range(NUM_MOVIES):
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

        ch_movies_data.append((mid, title, rating, rand_genre))
        movie_meta_map[mid] = {"rating": rating, "genre": rand_genre}

        # Случайный стартовый вектор
        vec = np.random.normal(0, 0.1, EMBEDDING_DIM)
        vec = (vec / np.linalg.norm(vec)).tolist()
        milvus_movies_data.append({
            "movie_id": mid,
            "title": title,
            "age_rating": rating,
            "embedding": vec
        })

    ch_client.execute("INSERT INTO movies_metadata (movie_id, title, age_rating, genre_type) VALUES", ch_movies_data)
    milvus_client.insert(collection_name="movies", data=milvus_movies_data)

    print("Генерация 50 000 аналитических логов просмотров в ClickHouse...")
    logs_data = []
    for _ in range(50000):
        uid = random.choice(list(user_age_map.keys()))
        mid = random.choice(list(movie_meta_map.keys()))
        user_age = user_age_map[uid]
        movie_genre = movie_meta_map[mid]["genre"]
        movie_rating = movie_meta_map[mid]["rating"]
        movie_len = random.randint(3600, 7200)

        if (user_age < 14 and movie_genre == 0) or \
                (14 <= user_age < 18 and movie_genre == 1) or \
                (user_age >= 18 and movie_genre == 2):
            watched = int(movie_len * random.uniform(0.7, 1.0))
        else:
            watched = int(movie_len * random.uniform(0.01, 0.15))

        if user_age < movie_rating:
            watched = random.randint(5, 60)

        logs_data.append((uid, mid, watched, movie_len))

    ch_client.execute(
        "INSERT INTO watch_logs (user_id, movie_id, watched_seconds, movie_duration_seconds) VALUES",
        logs_data
    )
    print("Базы данных успешно наполнены PRO-сценариями для Milvus Lite и ClickHouse!")


if __name__ == "__main__":
    init_databases()
    generate_and_insert_data()
