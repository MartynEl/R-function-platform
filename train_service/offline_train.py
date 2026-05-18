import os
import sys
from dotenv import load_dotenv
import psycopg2
from clickhouse_driver import Client
import torch
import torch.nn as nn
import torch.optim as optim

# Поднимаемся к корню для загрузки .env
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Параметры из .env
NUM_USERS = int(os.getenv("MODEL_NUM_USERS", 1000))
NUM_MOVIES = int(os.getenv("MODEL_NUM_ITEMS", 500))
EMBEDDING_DIM = int(os.getenv("MODEL_EMBEDDING_DIM", 16))


# --- 1. ПОДКЛЮЧЕНИЕ К БАЗАМ ---
def get_postgres_conn():
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


# --- 2. СЕТЕВАЯ АРХИТЕКТУРА ДЛЯ ОБУЧЕНИЯ ---
# (Должна быть идентична модели в FastAPI, чтобы веса подошли)
class RFunctionIntersection(nn.Module):
    def forward(self, x, y):
        return x + y - torch.sqrt(x ** 2 + y ** 2 + 1e-8)


class RFactorizationMachine(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.r_and = RFunctionIntersection()

    def forward(self, user_ids, item_ids, duration_ratios, constraint_g1, constraint_g2):
        u_b = self.user_bias(user_ids).squeeze(-1)
        i_b = self.item_bias(item_ids).squeeze(-1)
        u_e = self.user_emb(user_ids)
        i_e = self.item_emb(item_ids)

        interaction = torch.sum(u_e * i_e, dim=-1) * duration_ratios
        f_fm = u_b + i_b + interaction

        f_r = self.r_and(constraint_g1, constraint_g2)
        return self.r_and(f_fm, f_r)


# --- 3. ОСНОВНОЙ ЦИКЛ ОФЛАЙН ОБРАБОТКИ ---
def run_offline_training():
    print("=== Старт ночной аналитической задачи переобучения ===")

    # 1. Загружаем историю просмотров из ClickHouse
    print("Выгрузка поведенческих логов из ClickHouse...")
    ch_client = get_clickhouse_client()
    logs = ch_client.execute("SELECT user_id, movie_id, watched_seconds, movie_duration_seconds FROM watch_logs")

    if not logs:
        print("Логов в ClickHouse пока нет. Нечего обучать.")
        return

    print(f"Успешно получено {len(logs)} записей.")

    # 2. Загружаем метаданные из Postgres для формирования ограничений R-функции
    print("Синхронизация профилей пользователей и фильмов из PostgreSQL...")
    pg_conn = get_postgres_conn()
    with pg_conn.cursor() as cur:
        cur.execute("SELECT user_id, age FROM users")
        user_age_map = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("SELECT movie_id, age_rating FROM movies")
        movie_rating_map = {row[0]: row[1] for row in cur.fetchall()}

    # 3. Подготавливаем векторы (тензоры) для PyTorch
    users_t, items_t, durations_t, g1_t, g2_t = [], [], [], [], []

    for user_id, movie_id, watched, duration in logs:
        # Пропускаем, если данные в базах рассинхронизировались
        if user_id not in user_age_map or movie_id not in movie_rating_map:
            continue

        u_age = user_age_map[user_id]
        m_rating = movie_rating_map[movie_id]

        users_t.append(user_id)
        items_t.append(movie_id)
        # Считаем коэффициент удержания (длительность просмотра от 0.0 до 1.0)
        durations_t.append(float(watched / duration if duration > 0 else 0))
        g1_t.append(float(u_age - m_rating))
        g2_t.append(1.0)  # Второе ограничение всегда ок

    # Превращаем в тензоры CPU
    t_users = torch.tensor(users_t, dtype=torch.long)
    t_items = torch.tensor(items_t, dtype=torch.long)
    t_durations = torch.tensor(durations_t, dtype=torch.float)
    t_g1 = torch.tensor(g1_t, dtype=torch.float)
    t_g2 = torch.tensor(g2_t, dtype=torch.float)

    # 4. Инициализируем модель PyTorch и запускаем оптимизацию весов
    print("Инициализация градиентного спуска PyTorch...")
    model = RFactorizationMachine(NUM_USERS, NUM_MOVIES, EMBEDDING_DIM)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.03)
    criterion = nn.MSELoss()

    epochs = 40
    print(f"Запуск обучения (эпох: {epochs})...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(t_users, t_items, t_durations, t_g1, t_g2)

        # Таргет: если фильм безопасен, предсказываем duration (удержание).
        # Если нарушил ограничение Рвачёва (g1 < 0) — уводим таргет в -5.0
        target = torch.where((t_g1 >= 0) & (t_g2 >= 0), t_durations, torch.tensor(-5.0))

        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f" Эпоха {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    print("Обучение завершено. Экспорт весов в PostgreSQL...")
    model.eval()

    # 5. Сохраняем новые эмбеддинги в Postgres (в pgvector)
    # Извлекаем веса векторов из PyTorch модели
    user_embeddings = model.user_emb.weight.detach().numpy()
    movie_embeddings = model.item_emb.weight.detach().numpy()  # Матрица всех фильмов

    with pg_conn.cursor() as cur:
        # 1. Апдейтим веса фильмов
        for mid in range(NUM_MOVIES):
            cur.execute(
                "UPDATE movies SET embedding = %s WHERE movie_id = %s",
                (movie_embeddings[mid].tolist(), mid)
            )
        # 2. Апдейтим веса пользователей (Новая логика для Этапа А)
        for uid in range(NUM_USERS):
            cur.execute(
                "UPDATE users SET embedding = %s WHERE user_id = %s",
                (user_embeddings[uid].tolist(), uid)
            )

    pg_conn.commit()
    pg_conn.close()
    print("Эмбеддинги успешно записаны в pgvector! Ночные миграции завершены.")


if __name__ == "__main__":
    run_offline_training()
