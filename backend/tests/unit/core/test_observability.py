"""Logfire wiring with no token: the instrumentors are attached and spans are produced locally.

``capfire`` reconfigures Logfire with an in-memory exporter; the instrumentors attached at
app import keep working because they hold Logfire's proxy tracer provider. Nothing here
needs a ``LOGFIRE_TOKEN``, which is the point: the app must trace the same way it will in
production before anyone has an account.
"""

import json

from httpx import ASGITransport, AsyncClient
from logfire.testing import CaptureLogfire

from cortexdj.app import app


async def test_http_request_is_traced(capfire: CaptureLogfire) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/openapi.json")
    assert response.status_code == 200

    names = {span["name"] for span in capfire.exporter.exported_spans_as_dict()}
    assert "GET /api/openapi.json" in names


async def test_a_credential_query_param_never_reaches_a_request_span(capfire: CaptureLogfire) -> None:
    # The hook rides on the instrumentor attached at app import, so no reconfiguration is needed.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.get("/api/openapi.json", params={"token": "eyJ-secret-token", "q": "keep"})

    spans = capfire.exporter.exported_spans_as_dict()
    [request] = [span for span in spans if span["name"] == "GET /api/openapi.json"]
    assert "eyJ-secret-token" not in json.dumps(request["attributes"])
    assert request["attributes"]["http.url"] == "http://test/api/openapi.json?token=REDACTED&q=keep"
    assert request["attributes"]["logfire.msg"] == "GET /api/openapi.json ? q='keep' & token='REDACTED'"
