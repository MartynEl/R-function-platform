import os
import sys
import time
import random
import requests
import psycopg2
from dotenv import load_dotenv

# Поднимаемся к корню для загрузки .env
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))

API_URL = f"http://{os.getenv('SERVER_HOST', '127.0.0.1')}:{os.getenv('SERVER_PORT', '8080')}/predict"

def get_postgres_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB")
    )


def load_metadata_from_db():
    """ Сканируем базу данных, чтобы симулятор знал реальных юзеров и фильмы """
    conn = get_postgres_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT user_id, age FROM users;")
        users = cur.fetchall()  # Список кортежей (user_id, age)

        cur.execute("SELECT movie_id, age_rating, genre_type FROM movies;")
        movies = cur.fetchall()  # (movie_id, age_rating, genre_type)
    conn.close()
    return users, movies


def start_traffic_simulation():
    # ДЛИННАЯ ПАУЗА ПРИ СТАРТЕ: даем время FastAPI контейнеру и DNS сети Docker полностью подняться
    print("Ожидание инициализации сети Docker (5 секунд)...")
    time.sleep(5)

    print(f"Запуск внешнего генератора трафика. Шлём запросы на {API_URL}...")
    try:
        users, movies = load_metadata_from_db()
    except Exception as e:
        print(f"Ошибка подключения к PostgreSQL: {e}. Попробуем позже...")
        users, movies = [], []

    if not users or not movies:
        print("Предупреждение: База данных пуста или недоступна! Ждем 5 секунд перед повторной попыткой...")
        time.sleep(5)
        return  # Скрипт завершится, но Docker автоматически перезапустит его благодаря оркестрации

    try:
        while True:
            u_id, u_age = random.choice(users)
            m_id, m_rating, m_genre = random.choice(movies)

            if (u_age < 14 and m_genre == 0) or \
                    (14 <= u_age < 18 and m_genre == 1) or \
                    (u_age >= 18 and m_genre == 2):
                duration = random.uniform(0.75, 1.0)
            else:
                duration = random.uniform(0.02, 0.2)

            constraint_g1 = float(u_age - m_rating)
            constraint_g2 = 1.0

            payload = {
                "user_id": u_id,
                "item_id": m_id,
                "duration_ratio": round(duration, 2),
                "constraint_g1": constraint_g1,
                "constraint_g2": constraint_g2
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    print(
                        f"[Traffic Sim] User {u_id} (Age {u_age}) -> Movie {m_id} (Rating {m_rating}+). Status: {data['status']}. Score: {data['score']:.4f}")
                else:
                    print(f"[Traffic Sim] Ошибка API: {response.status_code} - {response.text}")

                # При успешном запросе делаем стандартную паузу в 0.5 сек
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                # ВАЖНО: Если сеть Docker еще «сырая», не спамим процессор, а спокойно ждем 2 секунды
                print(f"[Traffic Sim] Сервер API временно недоступен (ошибка сети). Ждем 2 сек...")
                time.sleep(2.0)

    except KeyboardInterrupt:
        print("\nГенерация трафика остановлена пользователем.")


if __name__ == "__main__":
    start_traffic_simulation()
