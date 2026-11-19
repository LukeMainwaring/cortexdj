"""Export the checked-out app's schema without a server, database, or provider calls.

Use fixed build-time settings so a developer's database and telemetry credentials
cannot affect generation. Importing the app does not run its lifespan hooks.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    """Write the SDK input beside the frontend generator, including on fresh clones."""
    with TemporaryDirectory(prefix="cortexdj-openapi-") as telemetry_dir:
        os.environ.update(
            API_PREFIX="/api",
            POSTGRES_USER="schema",
            POSTGRES_PASSWORD="schema",
            POSTGRES_DB="schema",
            POSTGRES_HOST="localhost",
            POSTGRES_PORT="5433",
            LOGFIRE_TOKEN="",
            LOGFIRE_DATA_DIR=telemetry_dir,
            LOGFIRE_CONSOLE="false",
        )
        from cortexdj.app import app

        output = Path(__file__).resolve().parents[2] / "frontend/api/scripts/outputs/openapi.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
        print(f"OpenAPI schema exported to {output}")


if __name__ == "__main__":
    main()
