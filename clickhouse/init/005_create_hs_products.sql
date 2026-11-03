-- Catálogo de productos del Sistema Armonizado (HS 1992) con embeddings.
-- Fuente: JSON de referencia de UNSD/Comtrade (H0.json, className HS1992).
-- El embedding se calcula antes del INSERT (nunca UPDATE), igual que en topics.
-- hs_code es String a propósito: los códigos HS empiezan con cero (por ejemplo 0101).
CREATE TABLE IF NOT EXISTS theme_curation.hs_products (
    hs_code         String,
    level           UInt8,            -- 2 capítulo, 4 partida, 6 subpartida
    parent_code     String,
    name            String,           -- descripción de UNSD sin el código y con marcadores :-- limpios
    chapter_code    String,
    chapter_name    String,
    heading_code    String,           -- vacío si level = 2
    heading_name    String,
    embed_context   String,           -- texto jerárquico que se vectoriza, sin el prefijo del modelo
    classification  String,           -- 'HS1992'
    embedding       Array(Float32),   -- 768 dimensiones, sin truncar
    model_id        String,
    created_at      DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (classification, level, hs_code);
