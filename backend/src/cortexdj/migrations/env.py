from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from cortexdj.core.db import get_postgres_url

config = context.config
# Doubled because the config file interpolates "%", and an encoded password can hold one.
config.set_main_option("sqlalchemy.url", get_postgres_url("postgresql+psycopg").replace("%", "%%"))

if config.config_file_name is not None:
    # Keep the app's loggers: the integration tier migrates in-process, and the default
    # would silence every logger already created, so no test could see a log line.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Importing the features package registers every feature's models on
# Base.metadata, so Alembic autogenerate sees all tables.
import cortexdj.features  # noqa: E402, F401
from cortexdj.shared.model_base import Base  # noqa: E402

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
