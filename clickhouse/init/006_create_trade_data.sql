-- Datos de comercio exterior del Atlas of Economic Complexity (Growth Lab, Harvard).
-- Fuente: hs92_country_product_year_4.csv o _6.csv (Harvard Dataverse, doi 10.7910/DVN/T4CHWJ).
-- Cita obligatoria: The Growth Lab at Harvard University, 2026, International Trade Data (HS92).
-- Valores en USD corrientes. Leer siempre con FINAL (ReplacingMergeTree).

-- Filas de los países que se carguen (por defecto Colombia, country_id 170).
CREATE TABLE IF NOT EXISTS theme_curation.trade_country_year (
    country_id           UInt32,
    country_iso3         String,
    hs_code              String,
    level                UInt8,              -- 4 o 6
    year                 UInt16,
    export_value         Int64,
    import_value         Int64,
    global_market_share  Float64,
    export_rca           Nullable(Float32),  -- solo en el archivo de 4 dígitos
    distance             Nullable(Float32),  -- solo en el archivo de 4 dígitos
    cog                  Nullable(Float32),  -- solo en el archivo de 4 dígitos
    pci                  Nullable(Float32),  -- solo en el archivo de 4 dígitos
    source_file          String,
    loaded_at            DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(loaded_at)
ORDER BY (country_id, level, hs_code, year);

-- Agregados mundiales por producto y año (calculados una sola vez sobre todos los países).
CREATE TABLE IF NOT EXISTS theme_curation.trade_world_year (
    hs_code              String,
    level                UInt8,
    year                 UInt16,
    world_export_value   Int64,
    n_exporters          UInt16,             -- países con exportaciones mayores que cero
    top_exporter_iso3    String,
    top_export_value     Int64,
    pci                  Nullable(Float32),
    source_file          String,
    loaded_at            DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(loaded_at)
ORDER BY (level, hs_code, year);
