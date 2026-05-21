import os
import sys
import time
import random
import requests
from clickhouse_driver import Client
from dotenv import load_dotenv

# Поднимаемся к корню для загрузки .env
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))

# Настраиваем URL на новый эндпоинт Умной Ленты (Smart Feed)
API_URL = f"http://{os.getenv('SERVER_HOST', '127.0.0.1')}:{os.getenv('SERVER_PORT', '8080')}/smart-feed"


def get_clickhouse_client():
    return Client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKHOUSE_PORT_NATIVE", 9000)),
        user=os.getenv("CLICKHOUSE_USER", "martin"),
        password=os.getenv("CLICKHOUSE_PASSWORD", "clickhouse_secure_pass_789"),
        database=os.getenv("CLICKHOUSE_DB", "r_analytics_db")
    )


def load_users_from_clickhouse():
    """Сканируем ClickHouse, чтобы симулятор знал сгенерированных пользователей и их реальный возраст"""
    ch_client = get_clickhouse_client()
    try:
        users = ch_client.execute("SELECT user_id, age FROM users_metadata")
        return users
    except Exception as e:
        print(f"[Traffic Sim] Ошибка чтения метаданных из ClickHouse: {e}")
        return []


def start_traffic_simulation():
    print("Ожидание инициализации сети Docker (5 секунд)...")
    time.sleep(5)
    print(f"Запуск генератора трафика v2.0-pro. Запросы на {API_URL}...")

    users = load_users_from_clickhouse()

    if not users:
        print("[Traffic Sim] Предупреждение: Таблица пользователей пуста. Ждем 5 секунд...")
        time.sleep(5)
        return

    try:
        while True:
            # Случайно выбираем пользователя из базы
            user_id, user_age = random.choice(users)

            # Формируем GET-запрос к умной ленте
            params = {
                "user_id": int(user_id),
                "current_age": int(user_age)
            }

            try:
                response = requests.get(API_URL, params=params, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    feed = data.get("feed", [])

                    if feed:
                        top_item = feed[0]
                        print(f"[Traffic Sim] User {user_id} (Age {user_age}) получил ленту из {len(feed)} фильмов. "
                              f"Топ-1: Movie {top_item['movie_id']} (Score: {top_item['score']}, Status: {top_item['status']})")
                    else:
                        print(f"[Traffic Sim] User {user_id} получил пустую ленту.")
                else:
                    print(f"[Traffic Sim] Ошибка API: {response.status_code} - {response.text}")

                # Пауза между запросами пользователей
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                print(f"[Traffic Sim] Сервер API временно недоступен. Ждем 2 сек...")
                time.sleep(2.0)

    except KeyboardInterrupt:
        print("\nГенерация трафика остановлена пользователем.")


if __name__ == "__main__":
    start_traffic_simulation()
