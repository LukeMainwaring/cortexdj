"""Prove the shipped migrations round-trip cleanly (downgrade base → head).

Safe only because the integration tier refuses to run against a database
whose name doesn't end with "_test" (see conftest ``_require_test_db``).
"""

from alembic import command
from alembic.config import Config

from tests.integration.conftest import _ALEMBIC_INI


def test_migrations_round_trip(_migrated: None) -> None:
    cfg = Config(_ALEMBIC_INI)
    try:
        command.downgrade(cfg, "base")
    finally:
        # Leave the schema at head for the rest of the tier regardless of outcome.
        command.upgrade(cfg, "head")
