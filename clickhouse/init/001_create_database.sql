-- Se ejecuta antes que los CREATE TABLE. Es redundante con CLICKHOUSE_DB del
-- docker-compose (que ya crea la base al iniciar el contenedor), pero se deja
-- explícito como red de seguridad: cada script de este directorio corre como
-- una invocación independiente de clickhouse-client, así que cada CREATE TABLE
-- referencia la base por nombre completo en vez de depender de un "USE"
-- previo que no persistiría entre archivos.
CREATE DATABASE IF NOT EXISTS theme_curation;
