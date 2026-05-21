{# Модель dbt для предобработки логов под R-функции #}
{{ config(materialized='view', engine='MergeTree') }}

SELECT
    w.user_id,
    w.movie_id,
    -- Вычисляем коэффициент удержания
    cast(w.watched_seconds as Float32) / cast(w.movie_duration_seconds as Float32) as duration_ratio,
    -- Предикат Рвачёва g1: разница возрастов для жесткого ограничения безопасности
    cast(u.age as Float32) - cast(m.age_rating as Float32) as constraint_g1
FROM {{ source('r_analytics_db', 'watch_logs') }} w
ANY LEFT JOIN {{ ref('users_metadata') }} u ON w.user_id = u.user_id
ANY LEFT JOIN {{ ref('movies_metadata') }} m ON w.movie_id = m.movie_id
