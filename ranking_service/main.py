import os
import sys
from dotenv import load_dotenv
import psycopg2

# Фиксируем окружение ОС до импорта torch
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import uvicorn
import logging
from clickhouse_driver import Client

# --- 1. НАСТРОЙКА ЛОГИРОВАНИЯ ---
logging.basicConfig(level=logging.INFO, format="INFO:     %(message)s")
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

    # СУБД параметры
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_user: str = "martin"
    postgres_password: str = "my_very_secure_password_123"
    postgres_db: str = "r_platform_db"

    clickhouse_host: str = "127.0.0.1"
    clickhouse_port_native: int = 9000
    clickhouse_user: str = "martin"
    clickhouse_password: str = "clickhouse_secure_pass_789"
    clickhouse_db: str = "r_analytics_db"


settings = ServerSettings()
device = torch.device(settings.model_device)


# --- 3. МАТЕМАТИКА РВАЧЁВА И МОДЕЛЬ ---
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


# Инициализируем каркас модели
model = RFactorizationMachine(
    num_users=settings.model_num_users,
    num_items=settings.model_num_items,
    embedding_dim=settings.model_embedding_dim
).to(device)


# --- 4. ПОДГРУЗКА ЭМБЕДДИНГОВ ИЗ PGVECTOR ---
def load_embeddings_from_postgres(model_instance):
    """ Подключается к Postgres и заменяет случайные веса фильма обученными векторами """
    logger.info("Синхронизация весов модели с pgvector из PostgreSQL...")
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        with conn.cursor() as cur:
            # Вытягиваем ID фильма и его текстовое представление вектора (pgvector возвращает '[x1,x2,...]')
            cur.execute("SELECT movie_id, embedding::text FROM movies WHERE embedding IS NOT NULL;")
            rows = cur.fetchall()

        if not rows:
            logger.info("В базе PostgreSQL пока нет обученных эмбеддингов. Используются случайные веса.")
            conn.close()
            return

        # Отключаем градиенты PyTorch для безопасного изменения весов «на лету»
        with torch.no_grad():
            loaded_count = 0
            for movie_id, emb_str in rows:
                if movie_id >= settings.model_num_items:
                    continue
                # Парсим строку '[0.1, 0.2, ...]' в список float
                emb_list = [float(x) for x in emb_str.strip("[]").split(",")]
                # Записываем тензор прямо в матрицу эмбеддингов модели
                model_instance.item_emb.weight[movie_id] = torch.tensor(emb_list, dtype=torch.float, device=device)
                loaded_count += 1

        logger.info(f"Синхронизация завершена! Успешно загружено {loaded_count} векторов из pgvector.")
        conn.close()
    except Exception as e:
        logger.error(f"Не удалось подтянуть веса из Postgres: {e}. Работаем на базовой инициализации.")


# Запускаем загрузку векторов в модель перед стартом API
load_embeddings_from_postgres(model)
model.eval()

# --- 5. ПОДКЛЮЧЕНИЕ К CLICKHOUSE ДЛЯ ОНЛАЙН-ЛОГОВ ---
ch_client = Client(
    host=settings.clickhouse_host,
    port=settings.clickhouse_port_native,
    user=settings.clickhouse_user,
    password=settings.clickhouse_password,
    database=settings.clickhouse_db
)


# --- 6. СХЕМЫ ДАННЫХ API ---
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


# --- 7. СЕРВИС FASTAPI ---
app = FastAPI(title="R-Function Live Recommendation Platform")


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

    return RecommendationResponse(
        user_id=request.user_id, item_id=request.item_id,
        score=score, is_safe=is_safe, status=status
    )


# --- ДОПОЛНИТЕЛЬНАЯ СХЕМА ОТВЕТА ДЛЯ ЭТАПА А ---
class CandidateMovie(BaseModel):
    movie_id: int
    title: str
    age_rating: int
    distance: float  # Косинусное расстояние (чем меньше, тем ближе вкус)


class TopRecommendationsResponse(BaseModel):
    user_id: int
    recommendations: list[CandidateMovie]


# --- НОВЫЙ ЭНДПОИНТ: ЭТАП А (CANDIDATE GENERATION) ---
@app.get("/recommend", response_model=TopRecommendationsResponse)
def get_top_candidates(user_id: int):
    """ Находит топ-20 лучших фильмов для юзера прямо внутри БД через pgvector """
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database=settings.postgres_db
        )
        with conn.cursor() as cur:
            # 1. Получаем вектор самого пользователя
            cur.execute("SELECT embedding FROM users WHERE user_id = %s AND embedding IS NOT NULL;", (user_id,))
            user_row = cur.fetchone()

            if not user_row:
                # Если пользователя нет или он еще не обучен (Cold Start)
                raise HTTPException(status_code=404, detail="Вектор пользователя не найден. Сначала обучите модель.")

            user_embedding_str = user_row[0]  # возвращает строку '[x1, x2, ...]'

            # 2. МАГИЯ PGVECTOR: Ищем топ-20 фильмов по косинусному расстоянию (<=>)
            cur.execute("""
                        SELECT movie_id, title, age_rating, (embedding <=> %s) as distance
                        FROM movies
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s
                            LIMIT 20;
                        """, (user_embedding_str, user_embedding_str))

            movie_rows = cur.fetchall()

        conn.close()

        # Формируем красивый список кандидатов
        candidates = [
            CandidateMovie(movie_id=row[0], title=row[1], age_rating=row[2], distance=float(row[3]))
            for row in movie_rows
        ]

        return TopRecommendationsResponse(user_id=user_id, recommendations=candidates)

    except Exception as e:
        logger.error(f"Ошибка Этапа А: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SmartFeedItem(BaseModel):
    movie_id: int
    title: str
    score: float
    status: str


class SmartFeedResponse(BaseModel):
    user_id: int
    feed: list[SmartFeedItem]


@app.get("/smart-feed", response_model=SmartFeedResponse)
def get_smart_feed(user_id: int, current_age: int):
    """
    ПОЛНЫЙ КОНВЕЙЕР:
    Этап А (pgvector) -> Передача кандидатов -> Этап Б (FM + R-функция Рвачёва)
    """
    try:
        # --- ШАГ 1: ЭТАП А (Вытаскиваем 20 кандидатов по вкусу из БД) ---
        conn = psycopg2.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database=settings.postgres_db
        )
        with conn.cursor() as cur:
            cur.execute("SELECT embedding FROM users WHERE user_id = %s AND embedding IS NOT NULL;", (user_id,))
            user_row = cur.fetchone()
            if not user_row:
                raise HTTPException(status_code=404, detail="Вектор пользователя не найден. Сначала обучите модель.")

            user_emb_str = user_row

            # Извлекаем кандидатов (берем чуть больше, например 30, так как R-функция часть отсеет)
            cur.execute("""
                        SELECT movie_id, title, age_rating
                        FROM movies
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s
                            LIMIT 30;
                        """, (user_emb_str,))
            candidates = cur.fetchall()
        conn.close()

        if not candidates:
            return SmartFeedResponse(user_id=user_id, feed=[])

        # --- ШАГ 2: ПОДГОТОВКА БАТЧА ДЛЯ ЭТАПА Б ---
        # Формируем списки для пакетной обработки в PyTorch (батчинг)
        movie_ids = [row[0] for row in candidates]
        titles = {row[0]: row[1] for row in candidates}
        ratings = {row[0]: row[2] for row in candidates}

        # Заполняем тензоры для всей пачки фильмов одновременно
        size = len(movie_ids)
        u_tensor = torch.tensor([user_id] * size, dtype=torch.long, device=device)
        i_tensor = torch.tensor(movie_ids, dtype=torch.long, device=device)
        d_tensor = torch.tensor([1.0] * size, dtype=torch.float,
                                device=device)  # Допустим, прогнозируем полный просмотр

        # Считаем ограничения Рвачёва для каждого фильма: g1 = Возраст пользователя - Ценз фильма
        g1_tensor = torch.tensor([float(current_age - ratings[mid]) for mid in movie_ids], dtype=torch.float,
                                 device=device)
        g2_tensor = torch.tensor([1.0] * size, dtype=torch.float, device=device)

        # --- ШАГ 3: ЭТАП Б (Плотное ранжирование + Рвачёв в PyTorch) ---
        with torch.no_grad():
            # Модель считает скоры для всех 30 фильмов ОДНИМ быстрым матричным ударом
            scores_tensor = model(u_tensor, i_tensor, d_tensor, g1_tensor, g2_tensor)
            scores = scores_tensor.cpu().numpy().tolist()

        # --- ШАГ 4: ФОРМИРОВАНИЕ И СОРТИРОВКА ЛЕНТЫ ---
        feed_items = []
        for mid, score in zip(movie_ids, scores):
            # Определяем статус на основе знака R-функции (если скор утянут глубоко вниз — значит Blocked)
            is_safe = current_age >= ratings[mid]
            status = "Approved" if is_safe else "Blocked by R-Function"

            feed_items.append(SmartFeedItem(
                movie_id=mid,
                title=titles[mid],
                score=round(score, 4),
                status=status
            ))

        # Сортируем ленту: Approved фильмы с высоким скором будут в самом верху, Blocked — улетят в конец
        feed_items.sort(key=lambda x: x.score, reverse=True)

        # Возвращаем топ-20 выживших и отсортированных фильмов
        return SmartFeedResponse(user_id=user_id, feed=feed_items[:20])

    except Exception as e:
        logger.error(f"Ошибка умной ленты: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.server_host, port=settings.server_port, reload=True)
