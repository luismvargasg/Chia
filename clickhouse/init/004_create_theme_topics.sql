-- Tópicos validados por eje temático (tabla asociativa, resuelve N:M entre
-- themes y topics). "Clasificación" en la UI.
--
-- ¡IMPORTANTE! ReplacingMergeTree no hace UPDATE in-place: reclasificar un
-- tópico inserta una fila nueva con classified_at más reciente, y la fila
-- vieja convive con la nueva hasta que ClickHouse las fusiona en segundo
-- plano (no es inmediato). Toda lectura desde el backend DEBE usar FINAL:
--   SELECT ... FROM theme_topics FINAL WHERE theme_id = ...
-- Omitir FINAL puede devolver duplicados/versiones viejas de forma
-- intermitente. A esta escala (miles de filas), el costo de FINAL es
-- insignificante — no hay excusa para omitirlo.
CREATE TABLE IF NOT EXISTS theme_curation.theme_topics (
    theme_id           String,
    topic_id           String,
    status              String,   -- 'included' | 'excluded' | 'borderline'
    similarity_score    Nullable(Float32),  -- null si el tópico se agregó manualmente
    method               String,   -- 'automatic' | 'manual'
    justification        String,   -- obligatoria si status = 'borderline'
    classified_at         DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(classified_at)
ORDER BY (theme_id, topic_id);
