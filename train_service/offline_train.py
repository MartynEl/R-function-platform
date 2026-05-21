import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from clickhouse_driver import Client
from pymilvus import MilvusClient
from dagster import asset, AssetExecutionContext, Definitions

# Поднимаемся к корню для загрузки .env
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Параметры из окружения
NUM_USERS = int(os.getenv("MODEL_NUM_USERS", 1000))
NUM_MOVIES = int(os.getenv("MODEL_NUM_ITEMS", 500))
EMBEDDING_DIM = int(os.getenv("MODEL_EMBEDDING_DIM", 16))
MILVUS_DB_PATH = "./milvus_pro_demo.db"


# --- 1. КЛИЕНТЫ СУБД ---
def get_clickhouse_client():
    return Client(
        host=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKHOUSE_PORT_NATIVE", 9000)),
        user=os.getenv("CLICKHOUSE_USER", "martin"),
        password=os.getenv("CLICKHOUSE_PASSWORD", "clickhouse_secure_pass_789"),
        database=os.getenv("CLICKHOUSE_DB", "r_analytics_db")
    )


def get_milvus_client():
    return MilvusClient(MILVUS_DB_PATH)


# --- 2. МАТЕМАТИЧЕСКАЯ АРХИТЕКТУРА (Идентична FastAPI) ---
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
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids, duration_ratios, constraint_g1, constraint_g2):
        u_b = self.user_bias(user_ids).squeeze(-1)
        i_b = self.item_bias(item_ids).squeeze(-1)
        u_e = self.user_emb(user_ids)
        i_e = self.item_emb(item_ids)
        interaction = torch.sum(u_e * i_e, dim=-1) * duration_ratios
        f_fm = u_b + i_b + interaction
        f_r = self.r_and(constraint_g1, constraint_g2)
        return self.r_and(f_fm, f_r)


# --- 3. ДЕКЛАРАТИВНЫЕ АССЕТЫ DAGSTER + ИМИТАЦИЯ dbt ---

@asset(description="Имитация dbt-модели stg_watch_features: подготовка датасета силами ClickHouse")
def dbt_stg_watch_features(context: AssetExecutionContext) -> list:
    """
    Здесь dbt в реальном продакшене выполнил бы SQL-запрос с Jinja шаблонами.
    ClickHouse мгновенно джоинит логи с метаданными, вычисляя g1 (R-предикат возраста).
    """
    context.log.info("Запуск dbt-модели: агрегация логов и расчет g1 в ClickHouse...")
    ch_client = get_clickhouse_client()

    # Тот самый dbt-подход: переносим логику маппинга из Python в СУБД
    dbt_query = """
                SELECT w.user_id, \
                       w.movie_id, \
                       w.watched_seconds / w.movie_duration_seconds           as duration_ratio, \
                       cast(u.age as Float32) - cast(m.age_rating as Float32) as g1
                FROM watch_logs w ANY LEFT JOIN users_metadata u \
                ON w.user_id = u.user_id
                    ANY LEFT JOIN movies_metadata m ON w.movie_id = m.movie_id \
                """
    # Если таблицы метаданных еще не перенесены в CH в вашем PoC, делаем безопасный фолбэк:
    try:
        logs = ch_client.execute(dbt_query)
    except Exception:
        context.log.warning("Таблицы метаданных в CH отсутствуют. Фолбэк на базовые логи.")
        raw_logs = ch_client.execute(
            "SELECT user_id, movie_id, watched_seconds, movie_duration_seconds FROM watch_logs")
        # Эмуляция dbt-трансформации на Python в случае отсутствия таблиц
        logs = [(r[0], r[1], float(r[2] / r[3] if r[3] > 0 else 0), 18.0 - 12.0) for r in raw_logs]

    context.log.info(f"dbt сформировал витрину данных. Получено {len(logs)} строк.")
    return logs


@asset(description="Обучение FM-модели на подготовленном dbt-ассете с учетом R-функций")
def trained_pytorch_embeddings(context: AssetExecutionContext, dbt_stg_watch_features: list) -> dict:
    if not dbt_stg_watch_features:
        context.log.error("Нет данных для обучения от dbt.")
        return {}

    context.log.info("Инициализация вычислительного графа PyTorch...")
    users_t, items_t, durations_t, g1_t, g2_t = [], [], [], [], []

    for user_id, movie_id, duration_ratio, g1 in dbt_stg_watch_features:
        users_t.append(user_id)
        items_t.append(movie_id)
        durations_t.append(duration_ratio)
        g1_t.append(g1)
        g2_t.append(1.0)

    t_users = torch.tensor(users_t, dtype=torch.long)
    t_items = torch.tensor(items_t, dtype=torch.long)
    t_durations = torch.tensor(durations_t, dtype=torch.float)
    t_g1 = torch.tensor(g1_t, dtype=torch.float)
    t_g2 = torch.tensor(g2_t, dtype=torch.float)

    model = RFactorizationMachine(NUM_USERS, NUM_MOVIES, EMBEDDING_DIM)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    criterion = nn.MSELoss()

    epochs = 40
    context.log.info(f"Старт градиентного спуска. Эпох: {epochs}...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(t_users, t_items, t_durations, t_g1, t_g2)
        target = torch.where((t_g1 >= 0) & (t_g2 >= 0), t_durations, torch.tensor(-5.0))
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()

    context.log.info(f"Обучение завершено. Финальный Loss: {loss.item():.4f}")
    model.eval()

    return {
        "user_embeddings": model.user_emb.weight.detach().numpy(),
        "movie_embeddings": model.item_emb.weight.detach().numpy()
    }


@asset(description="Экспорт весов в коллекции Milvus Lite для инференса")
def milvus_vector_storage_sync(context: AssetExecutionContext, trained_pytorch_embeddings: dict):
    if not trained_pytorch_embeddings:
        context.log.error("Экспорт отменен: отсутствуют обученные эмбеддинги.")
        return

    context.log.info("Подключение к Milvus Lite...")
    milvus_client = get_milvus_client()

    user_embs = trained_pytorch_embeddings["user_embeddings"]
    movie_embs = trained_pytorch_embeddings["movie_embeddings"]

    # Пакетный апдейт пользователей в Milvus через .upsert
    user_data = [{"user_id": uid, "embedding": user_embs[uid].tolist()} for uid in range(NUM_USERS)]
    milvus_client.upsert(collection_name="users", data=user_data)
    context.log.info(f"Успешно синхронизировано {NUM_USERS} векторов пользователей в Milvus.")

    # Пакетный апдейт фильмов в Milvus
    # Для демонстрации обновляем только векторы (FastAPI при поиске довытащит метаданные)
    movie_data = []
    for mid in range(NUM_MOVIES):
        movie_data.append({
            "movie_id": mid,
            "embedding": movie_embs[mid].tolist(),
            "title": f"Фильм {mid}",  # Дефолтные значения на случай первой инициализации
            "age_rating": 12 if mid % 2 == 0 else 18
        })
    milvus_client.upsert(collection_name="movies", data=movie_data)
    context.log.info(f"Успешно синхронизировано {NUM_MOVIES} векторов фильмов в Milvus.")
    context.log.info("Ночной пайплайн Dagster + Milvus полностью выполнен!")


# --- 4. ОПРЕДЕЛЕНИЕ РЕПОЗИТОРИЯ DAGSTER ---
# Это заменяет ручной запуск функций и cron. Дагстер сам строит граф зависимостей.
defs = Definitions(
    assets=[dbt_stg_watch_features, trained_pytorch_embeddings, milvus_vector_storage_sync]
)

if __name__ == "__main__":
    # Код для запуска через консоль, чтобы Docker-cron продолжал работать без изменений
    from dagster import materialize

    materialize([dbt_stg_watch_features, trained_pytorch_embeddings, milvus_vector_storage_sync])
