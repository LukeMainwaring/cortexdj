"""The shipped migration upgrades AND downgrades cleanly (round-trip)."""

from alembic import command
from alembic.config import Config

from tests.integration.conftest import ALEMBIC_INI


def test_migration_downgrade_upgrade_round_trip() -> None:
    # `downgrade base` DROPS ALL TABLES; safe because the whole integration tier refuses to run
    # outside an isolated test database (see _require_test_db in conftest.py).
    cfg = Config(ALEMBIC_INI)
    try:
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")
    finally:
        command.upgrade(cfg, "head")  # leave schema present for the rest of the session
