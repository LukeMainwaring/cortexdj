"""Database queries on the app's engine are traced.

Uses ``core.db``'s engine itself (not the conftest's test engine) because that is the
engine ``setup_observability`` instruments; it only reads, so nothing leaks.
"""

from logfire.testing import CaptureLogfire
from sqlalchemy import text

from cortexdj.core.db import async_engine


async def test_database_queries_are_traced(capfire: CaptureLogfire) -> None:
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    finally:
        await async_engine.dispose()  # leave no pooled connection behind in the test process

    statements = [span["attributes"].get("db.statement") for span in capfire.exporter.exported_spans_as_dict()]
    assert "SELECT 1" in statements
