"""Request logging keeps credential query parameters out of the logs."""

import logging

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.datastructures import URL, QueryParams

from cortexdj.app import app
from cortexdj.core.logging import redact_query_params, redact_url


async def test_the_request_log_redacts_a_token_query_param(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="cortexdj.app")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.get("/api/openapi.json", params={"token": "eyJ-secret-token"})

    [request_line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Request:")]
    assert request_line == "Request: GET http://test/api/openapi.json?token=REDACTED"


def test_redacting_query_params_keeps_every_other_param() -> None:
    params = QueryParams("q=budget&kind=policy&kind=product&token=eyJ-secret-token")

    assert str(redact_query_params(params)) == "q=budget&kind=policy&kind=product&token=REDACTED"


def test_credential_keys_match_in_any_spelling() -> None:
    params = QueryParams("apiKey=k1&refresh-token=k2&Token=k3&client_secret=k4&password=k5&sig=k6&q=keep")

    assert str(redact_query_params(params)) == (
        "apiKey=REDACTED&refresh-token=REDACTED&Token=REDACTED&client_secret=REDACTED"
        "&password=REDACTED&sig=REDACTED&q=keep"
    )


def test_a_url_with_nothing_to_redact_is_logged_as_sent() -> None:
    url = URL("http://test/api/notes/search?q=a%20b%2Bc")

    assert redact_url(url) == "http://test/api/notes/search?q=a%20b%2Bc"


def test_the_uvicorn_access_log_redacts_a_token_query_param(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.access")
    # The call uvicorn's HTTP protocols make for each response.
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET", "/api/stream?token=eyJ-secret-token&q=keep", "1.1", 200
    )

    assert caplog.records[-1].getMessage() == '127.0.0.1:5000 - "GET /api/stream?token=REDACTED&q=keep HTTP/1.1" 200'
