"""Prove the shipped migrations round-trip cleanly (downgrade base → head).

Safe only because the integration tier refuses to run against a database
whose name lacks "test" (see conftest ``_require_test_db``).
"""

import pytest
from alembic import command
from alembic.config import Config

pytestmark = pytest.mark.integration

_ALEMBIC_INI = "src/cortexdj/alembic.ini"


def test_migrations_round_trip(_migrated: None) -> None:
    cfg = Config(_ALEMBIC_INI)
    try:
        command.downgrade(cfg, "base")
    finally:
        # Leave the schema at head for the rest of the tier regardless of outcome.
        command.upgrade(cfg, "head")
