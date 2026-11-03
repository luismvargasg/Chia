-- Catálogo de tópicos de OpenAlex + embeddings precalculados (carga única).
-- Los embeddings se generan en el script de carga (Ollama; Qwen3-Embedding-4B
-- sin prefijo, o Nomic con "search_document:") ANTES del INSERT — nunca se hace UPDATE sobre esta
-- tabla, porque en ClickHouse (MergeTree) los UPDATE son mutaciones
-- asíncronas y costosas, no operaciones in-place.
CREATE TABLE IF NOT EXISTS theme_curation.topics (
    topic_id       String,
    name           String,
    summary        String,
    keywords       String,
    subfield_id    String,
    subfield_name  String,
    field_id       String,
    field_name     String,
    domain_id      String,   -- jerarquía propia de OpenAlex, NO confundir con "theme"
    domain_name    String,
    wikipedia_url  String,
    embedding      Array(Float32),  -- sin longitud fija: 2560 (Qwen3-Embedding-4B, desde D-061) o 768 (Nomic, antes)
    model_id       String,          -- ej. 'qwen3-embedding:4b' (antes 'nomic-embed-text-v2-moe')
    created_at     DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY topic_id;
