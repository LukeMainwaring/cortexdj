"""The /agent/chat persistence contract: history lives server-side and survives a broken run.

The client sends only its newest message (see ``frontend/components/chat.tsx``), so
anything the server fails to store is gone for good — these cover both halves of that
bargain. The model is one of the chat conftest's stream doubles.
"""

import json
import logging
from collections.abc import AsyncIterator

import pytest
from httpx import AsyncClient
from pydantic_ai.exceptions import (
    ModelAuthenticationError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelNotFoundError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestRejectedError,
)
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from cortexdj.agents.registry import ASSISTANT, get_agent
from tests.factories import make_chat_request
from tests.integration.features.chat.conftest import CHAT_URL, PromptRecorder, die_mid_stream, reply_ok


async def _roles(client: AsyncClient, thread_id: str) -> list[str]:
    response = await client.get(f"/api/threads/{thread_id}/messages")
    return [message["role"] for message in response.json()["messages"]]


def _error_text(response_text: str) -> str:
    chunks = [
        json.loads(line.removeprefix("data: ")) for line in response_text.splitlines() if line.startswith("data: {")
    ]
    [error] = [chunk for chunk in chunks if chunk["type"] == "error"]
    error_text: str = error["errorText"]
    return error_text


async def test_second_turn_is_answered_from_stored_history(client: AsyncClient) -> None:
    recorder = PromptRecorder()
    with get_agent(ASSISTANT).override(model=recorder.model()):
        assert (await client.post(CHAT_URL, json=make_chat_request("t-turns", "first"))).status_code == 200
        assert (await client.post(CHAT_URL, json=make_chat_request("t-turns", "second"))).status_code == 200

    # Turn 2 carried only its own message, yet the run saw turn 1 — read back from the database.
    assert recorder.prompts == [["first"], ["first", "second"]]

    assert await _roles(client, "t-turns") == ["user", "assistant", "user", "assistant"]


async def test_a_malformed_body_is_the_apps_usual_422_not_a_server_error(client: AsyncClient) -> None:
    """Body validation fails before anything is stored, in the `detail` shape every other route uses."""
    response = await client.post(CHAT_URL, json={"trigger": "submit-message", "messages": "not a list"})

    assert response.status_code == 422
    [error] = [e for e in response.json()["detail"] if e["loc"][-1] == "messages"]
    assert error["loc"][0] == "body"
    assert "input" not in error, "the raw turn must not be echoed back"


async def test_a_body_that_is_not_utf8_is_still_a_422(client: AsyncClient) -> None:
    """Pydantic's error carries the raw bytes as `input`; left in, FastAPI's encoder fails and the client sees a 500."""
    response = await client.post(CHAT_URL, content=b"\xff\xfe{", headers={"content-type": "application/json"})

    assert response.status_code == 422


async def test_a_non_json_content_type_is_refused_before_the_body_is_read(client: AsyncClient) -> None:
    response = await client.post(CHAT_URL, content=b"hello", headers={"content-type": "text/plain"})

    assert response.status_code == 415


async def test_a_turn_stored_twice_reads_back_under_two_different_ids(client: AsyncClient) -> None:
    """``stream.py``'s accepted Retry: the reply was stored, so the retried turn is stored again.

    Both stored copies carry the same client message id on the wire; the thread must not read
    back with two messages under one id, which the frontend keys on.
    """
    turn = make_chat_request("t-same-id", "again please")
    with get_agent(ASSISTANT).override(model=FunctionModel(stream_function=reply_ok)):
        assert (await client.post(CHAT_URL, json=turn)).status_code == 200
        assert (await client.post(CHAT_URL, json={**turn, "trigger": "regenerate-message"})).status_code == 200

    response = await client.get("/api/threads/t-same-id/messages")
    ids = [message["id"] for message in response.json()["messages"]]
    assert await _roles(client, "t-same-id") == ["user", "assistant", "user", "assistant"]
    assert len(set(ids)) == len(ids), ids


async def test_broken_run_still_stores_the_user_turn(client: AsyncClient) -> None:
    # The adapter reports a run error as an SSE error event, so the request itself still succeeds.
    with get_agent(ASSISTANT).override(model=FunctionModel(stream_function=die_mid_stream)):
        assert (await client.post(CHAT_URL, json=make_chat_request("t-broken", "keep me"))).status_code == 200

    # `on_complete` never fired, but the user's message is stored: the client won't re-send it.
    assert await _roles(client, "t-broken") == ["user"]


async def test_regenerating_the_unanswered_turn_after_a_broken_run_does_not_store_it_twice(
    client: AsyncClient,
) -> None:
    recorder = PromptRecorder()
    turn = make_chat_request("t-retry", "Yes, let's do that")
    with get_agent(ASSISTANT).override(model=FunctionModel(stream_function=die_mid_stream)):
        await client.post(CHAT_URL, json=turn)
    with get_agent(ASSISTANT).override(model=recorder.model()):
        retry = await client.post(CHAT_URL, json={**turn, "trigger": "regenerate-message"})
        again = await client.post(CHAT_URL, json={**turn, "trigger": "regenerate-message"})

    assert (retry.status_code, again.status_code) == (200, 200)
    assert recorder.prompts == [["Yes, let's do that"], ["Yes, let's do that", "Yes, let's do that"]]
    # Once answered, the same words are a new turn again, even on a regenerate.
    assert await _roles(client, "t-retry") == ["user", "assistant", "user", "assistant"]


async def test_submitting_the_same_words_again_is_a_new_turn_even_when_the_last_went_unanswered(
    client: AsyncClient,
) -> None:
    """A run still in flight leaves history ending on the user's turn too, so a submit is never read as a retry."""
    with get_agent(ASSISTANT).override(model=FunctionModel(stream_function=die_mid_stream)):
        await client.post(CHAT_URL, json=make_chat_request("t-twice", "hi"))
    with get_agent(ASSISTANT).override(model=FunctionModel(stream_function=reply_ok)):
        await client.post(CHAT_URL, json=make_chat_request("t-twice", "hi"))

    assert await _roles(client, "t-twice") == ["user", "user", "assistant"]


async def test_a_run_that_dies_mid_stream_is_logged_and_the_error_reaches_the_client(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="cortexdj.features.chat.stream")
    with get_agent(ASSISTANT).override(model=FunctionModel(stream_function=die_mid_stream)):
        response = await client.post(CHAT_URL, json=make_chat_request("t-logged", "hi"))

    [record] = [r for r in caplog.records if r.name == "cortexdj.features.chat.stream"]
    assert "Agent run ended with an error mid-stream" in record.getMessage()
    assert record.exc_info is not None  # logged with the traceback
    # The parent's error chunk still went to the browser: the client sees a failed turn, not a clean finish.
    assert '"type":"error"' in response.text.replace(" ", "")


@pytest.mark.parametrize(
    ("category", "status_code", "detail"),
    [
        (ModelRateLimitError, 429, "The model provider is busy right now. Try again in a moment."),
        # Overloaded is a kind of unavailable, so it gets the same message.
        (ModelOverloadedError, 529, "The model provider is busy right now. Try again in a moment."),
        (
            ModelContextWindowExceededError,
            400,
            "This conversation is too long for the model. Start a new chat to keep going.",
        ),
        (
            ModelQuotaExceededError,
            402,
            "The model provider's quota or credits are used up. Retrying won't help until it is topped up.",
        ),
        (
            ModelAuthenticationError,
            401,
            "The model provider rejected the server's credentials or their access to this model. "
            "Retrying won't help until that is fixed.",
        ),
        (
            ModelNotFoundError,
            404,
            "The model provider couldn't find the configured model, or the server's provider URL is wrong. "
            "Retrying won't help until that is fixed.",
        ),
        # Any other rejection: the catch-all row, after the categories that subclass it.
        (ModelRequestRejectedError, 422, "The model provider refused this request. Retrying it unchanged won't help."),
    ],
)
async def test_a_known_provider_failure_reaches_the_client_as_a_message_to_show(
    client: AsyncClient,
    category: type[
        ModelRateLimitError
        | ModelOverloadedError
        | ModelContextWindowExceededError
        | ModelQuotaExceededError
        | ModelAuthenticationError
        | ModelNotFoundError
        | ModelRequestRejectedError
    ],
    status_code: int,
    detail: str,
) -> None:
    async def reject(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise ModelHTTPError.for_category(category, status_code=status_code, model_name="function")
        yield  # an async generator, so the stream fails on its first read

    with get_agent(ASSISTANT).override(model=FunctionModel(stream_function=reject)):
        response = await client.post(CHAT_URL, json=make_chat_request("t-known", "hi"))

    assert json.loads(_error_text(response.text)) == {"detail": detail}


async def test_an_unrecognized_failure_keeps_its_raw_text_for_the_generic_fallback(client: AsyncClient) -> None:
    with get_agent(ASSISTANT).override(model=FunctionModel(stream_function=die_mid_stream)):
        response = await client.post(CHAT_URL, json=make_chat_request("t-unknown", "hi"))

    # Not a JSON detail, so the browser shows "The assistant could not reply. Try again."
    assert _error_text(response.text) == "provider is down"
