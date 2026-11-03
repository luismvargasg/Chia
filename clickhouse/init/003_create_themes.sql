-- Ejes temáticos curados. En UI: "Eje Temático". En código/BD: "theme"
-- (evita colisión con domain_id/domain_name, que ya pertenece a la
-- jerarquía propia de OpenAlex en la tabla topics).
CREATE TABLE IF NOT EXISTS theme_curation.themes (
    theme_id       String,
    name           String,
    description    String,   -- descripción pública, distinta de la semilla
    seed_text      String,
    curator        String,   -- texto libre, no hay cuentas de usuario
    created_at     DateTime DEFAULT now(),
    updated_at     DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY theme_id;
