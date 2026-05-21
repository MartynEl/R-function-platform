import os
import sys
import logging
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import uvicorn
from clickhouse_driver import Client
from dotenv import load_dotenv

# ИМПОРТ МИЛВУСА
from pymilvus import MilvusClient

# Фиксируем окружение ОС до импорта torch
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- 1. НАСТРОЙКА ЛОГИРОВАНИЯ ---
logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
logger = logging.getLogger("app")


# --- 2. НАСТРОЙКИ СЕРВЕРА ---
class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    model_num_users: int = 1000
    model_num_items: int = 500
    model_embedding_dim: int = 16
    model_device: str = "cpu"
    server_host: str = "127.0.0.1"
    server_port: int = 8080

    # СУБД параметры (Postgres полностью убран)
    clickhouse_host: str = "127.0.0.1"
    clickhouse_port_native: int = 9000
    clickhouse_user: str = "martin"
    clickhouse_password: str = "clickhouse_secure_pass_789"
    clickhouse_db: str = "r_analytics_db"

    # Путь к локальному файлу базы Milvus Lite
    milvus_db_path: str = "./milvus_pro_demo.db"


settings = ServerSettings()
device = torch.device(settings.model_device)

# --- 3. ИНИЦИАЛИЗАЦИЯ MILVUS LITE С КОРРЕКТНЫМ СИНТАКСИСОМ ---
milvus_client = MilvusClient(settings.milvus_db_path)

# Автоматически создаем коллекции, если сервер стартует в чистом окружении
if not milvus_client.has_collection("users"):
    schema_u = MilvusClient.create_schema(auto_id=False)
    schema_u.add_field(field_name="user_id", datatype=DataType.INT64, is_primary=True)
    schema_u.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=settings.model_embedding_dim)
    milvus_client.create_collection(collection_name="users", schema=schema_u)

if not milvus_client.has_collection("movies"):
    schema_m = MilvusClient.create_schema(auto_id=False)
    schema_m.add_field(field_name="movie_id", datatype=DataType.INT64, is_primary=True)
    schema_m.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=255)
    schema_m.add_field(field_name="age_rating", datatype=DataType.INT64)
    schema_m.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=settings.model_embedding_dim)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name="embedding", metric_type="COSINE", index_type="FLAT")
    milvus_client.create_collection(collection_name="movies", schema=schema_m, index_params=index_params)

milvus_client.load_collection("users")
milvus_client.load_collection("movies")
logger.info("Коллекции Milvus успешно загружены в оперативную память!")

# --- 4. МАТЕМАТИКА РВАЧЁВА И МОДЕЛЬ ---
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
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
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
        f_total = self.r_and(f_fm, f_r)
        return f_total


model = RFactorizationMachine(
    num_users=settings.model_num_users,
    num_items=settings.model_num_items,
    embedding_dim=settings.model_embedding_dim
).to(device)


# --- 5. СИНХРОНИЗАЦИЯ МОДЕЛИ С MILVUS LITE ---
def load_embeddings_from_milvus(model_instance):
    """ Накатывает обученные векторы из Milvus Lite в инференс-модель PyTorch """
    logger.info("Синхронизация весов модели с хранилищем Milvus Lite...")
    try:
        with torch.no_grad():
            # Загружаем фильмы
            movie_res = milvus_client.query(collection_name="movies", filter="movie_id >= 0",
                                            output_fields=["movie_id", "embedding"], limit=settings.model_num_items)
            for item in movie_res:
                mid = item["movie_id"]
                if "embedding" in item and mid < settings.model_num_items:
                    model_instance.item_emb.weight[mid] = torch.tensor(item["embedding"], dtype=torch.float,
                                                                       device=device)

            # Загружаем пользователей
            user_res = milvus_client.query(collection_name="users", filter="user_id >= 0",
                                           output_fields=["user_id", "embedding"], limit=settings.model_num_users)
            for item in user_res:
                uid = item["user_id"]
                if "embedding" in item and uid < settings.model_num_users:
                    model_instance.user_emb.weight[uid] = torch.tensor(item["embedding"], dtype=torch.float,
                                                                       device=device)
            logger.info("Синхронизация с Milvus завершена успешно!")
    except Exception as e:
        logger.error(f"Не удалось подтянуть веса из Milvus: {e}. Работаем на базовой инициализации.")


load_embeddings_from_milvus(model)
model.eval()

# --- 6. ПОДКЛЮЧЕНИЕ К CLICKHOUSE ДЛЯ ОНЛАЙН-ЛОГОВ ---
ch_client = Client(
    host=settings.clickhouse_host,
    port=settings.clickhouse_port_native,
    user=settings.clickhouse_user,
    password=settings.clickhouse_password,
    database=settings.clickhouse_db
)


# --- 7. СХЕМЫ ДАННЫХ API ---
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    item_id: int = Field(..., ge=0)
    duration_ratio: float = Field(0.0, ge=0.0, le=1.0)
    constraint_g1: float = Field(...)
    constraint_g2: float = Field(...)


class RecommendationResponse(BaseModel):
    user_id: int
    item_id: int
    score: float
    is_safe: bool
    status: str


def log_watch_event_to_clickhouse(user_id: int, movie_id: int, duration_ratio: float):
    try:
        movie_duration = 5400
        watched_seconds = int(movie_duration * duration_ratio)
        ch_client.execute(
            "INSERT INTO watch_logs (user_id, movie_id, watched_seconds, movie_duration_seconds) VALUES",
            [(user_id, movie_id, watched_seconds, movie_duration)]
        )
    except Exception as e:
        logger.error(f"Не удалось записать лог в ClickHouse: {e}")


# --- 8. СЕРВИС FASTAPI ---
app = FastAPI(title="R-Function Live Recommendation Platform [PRO VERSION]")


@app.post("/predict", response_model=RecommendationResponse)
def predict_score(request: RecommendationRequest, background_tasks: BackgroundTasks):
    if request.user_id >= settings.model_num_users or request.item_id >= settings.model_num_items:
        raise HTTPException(status_code=400, detail="User ID or Item ID out of bounds.")

    u_tensor = torch.tensor([request.user_id], dtype=torch.long, device=device)
    i_tensor = torch.tensor([request.item_id], dtype=torch.long, device=device)
    d_tensor = torch.tensor([request.duration_ratio], dtype=torch.float, device=device)
    g1_tensor = torch.tensor([request.constraint_g1], dtype=torch.float, device=device)
    g2_tensor = torch.tensor([request.constraint_g2], dtype=torch.float, device=device)

    with torch.no_grad():
        score_tensor = model(u_tensor, i_tensor, d_tensor, g1_tensor, g2_tensor)
        score = float(score_tensor.item())

    is_safe = request.constraint_g1 >= 0 and request.constraint_g2 >= 0
    status = "Approved" if is_safe else "Blocked by R-Function Constraint"

    background_tasks.add_task(log_watch_event_to_clickhouse, request.user_id, request.item_id, request.duration_ratio)
    return RecommendationResponse(user_id=request.user_id, item_id=request.item_id, score=score, is_safe=is_safe,
                                  status=status)


class CandidateMovie(BaseModel):
    movie_id: int
    title: str
    age_rating: int
    distance: float


class TopRecommendationsResponse(BaseModel):
    user_id: int
    recommendations: list[CandidateMovie]


# --- ЭТАП А: ОНЛАЙН-ПОИСК КАНДИДАТОВ ЧЕРЕЗ MILVUS LITE ---
@app.get("/recommend", response_model=TopRecommendationsResponse)
def get_top_candidates(user_id: int):
    try:
        user_res = milvus_client.get(collection_name="users", ids=[user_id], output_fields=["embedding"])
        if not user_res:
            raise HTTPException(status_code=404, detail="Вектор пользователя не найден в Milvus.")

        # Безопасное извлечение вектора (обработка списков и словарей от Milvus Client)
        if isinstance(user_res, list) and len(user_res) > 0:
            user_vector = user_res[0]["embedding"]
        else:
            user_vector = user_res["embedding"]

        # Нативный векторный поиск по индексу COSINE
        search_res = milvus_client.search(
            collection_name="movies", data=[user_vector], limit=20,
            output_fields=["title", "age_rating", "movie_id"]
        )

        candidates = []
        if search_res:
            # MilvusClient возвращает список батчей, берем первый батч результатов
            hits = search_res[0] if isinstance(search_res[0], list) else search_res
            for hit in hits:
                entity = hit.get("entity", {})
                m_id = entity.get("movie_id", hit.get("id"))
                candidates.append(CandidateMovie(
                    movie_id=int(m_id),
                    title=entity.get("title", f"Фильм {m_id}"),
                    age_rating=entity.get("age_rating", 12),
                    distance=float(hit.get("distance", 0.0))
                ))
        return TopRecommendationsResponse(user_id=user_id, recommendations=candidates)
    except Exception as e:
        logger.error(f"Ошибка Этапа А в Milvus: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- КЛАССЫ СХЕМ ДАННЫХ ДЛЯ СМАРТ-ЛЕНТЫ ---
class SmartFeedItem(BaseModel):
    movie_id: int
    title: str
    score: float
    status: str


class SmartFeedResponse(BaseModel):
    user_id: int
    feed: list[SmartFeedItem]


# --- ПОЛНЫЙ ДВУХЭТАПНЫЙ КОНВЕЙЕР (ЭТАП А В MILVUS -> ЭТАП Б В PYTORCH С R-ФУНКЦИЕЙ) ---
@app.get("/smart-feed", response_model=SmartFeedResponse)
def get_smart_feed(user_id: int, current_age: int):
    try:
        # 1. Извлекаем вектор пользователя
        user_res = milvus_client.get(collection_name="users", ids=[user_id], output_fields=["embedding"])
        if not user_res:
            raise HTTPException(status_code=404, detail="Вектор пользователя не найден в Milvus.")

        if isinstance(user_res, list) and len(user_res) > 0:
            user_vector = user_res[0]["embedding"]
        else:
            user_vector = user_res["embedding"]

        # 2. Извлекаем 30 кандидатов по вкусу через Milvus
        search_res = milvus_client.search(
            collection_name="movies", data=[user_vector], limit=30,
            output_fields=["title", "age_rating", "movie_id"]
        )

        if not search_res or len(search_res) == 0:
            return SmartFeedResponse(user_id=user_id, feed=[])

        hits = search_res[0] if isinstance(search_res[0], list) else search_res
        if not hits:
            return SmartFeedResponse(user_id=user_id, feed=[])

        # 3. Батчинг параметров для PyTorch модели (Этап Б) с учетом декларативного movie_id
        movie_ids = []
        titles = {}
        ratings = {}

        for hit in hits:
            entity = hit.get("entity", {})
            m_id = entity.get("movie_id", hit.get("id"))
            if m_id is not None:
                m_id = int(m_id)
                movie_ids.append(m_id)
                titles[m_id] = entity.get("title", f"Фильм {m_id}")
                ratings[m_id] = entity.get("age_rating", 12)

        size = len(movie_ids)
        if size == 0:
            return SmartFeedResponse(user_id=user_id, feed=[])

        u_tensor = torch.tensor([user_id] * size, dtype=torch.long, device=device)
        i_tensor = torch.tensor(movie_ids, dtype=torch.long, device=device)
        d_tensor = torch.tensor([1.0] * size, dtype=torch.float, device=device)

        # Ограничение R-функции: g1 = Возраст пользователя - Ценз фильма
        g1_tensor = torch.tensor([float(current_age - ratings[mid]) for mid in movie_ids], dtype=torch.float,
                                 device=device)
        g2_tensor = torch.tensor([1.0] * size, dtype=torch.float, device=device)

        # 4. Скоринг и фильтрация Рвачёва в один матричный проход
        with torch.no_grad():
            scores_tensor = model(u_tensor, i_tensor, d_tensor, g1_tensor, g2_tensor)
            scores = scores_tensor.cpu().numpy().tolist()

        # 5. Сортировка выдачи
        feed_items = []
        for mid, score in zip(movie_ids, scores):
            is_safe = current_age >= ratings[mid]
            status = "Approved" if is_safe else "Blocked by R-Function"

            feed_items.append(SmartFeedItem(
                movie_id=mid, title=titles[mid], score=round(score, 4), status=status
            ))

        feed_items.sort(key=lambda x: x.score, reverse=True)
        return SmartFeedResponse(user_id=user_id, feed=feed_items[:20])
    except Exception as e:
        logger.error(f"Ошибка умной ленты: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.server_host, port=settings.server_port, reload=True)
