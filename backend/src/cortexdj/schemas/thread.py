from datetime import datetime
from typing import Any

from pydantic import Field

from cortexdj.schemas.agent_type import AgentType

from .base import BaseSchema


class BrainContext(BaseSchema):
    latest_session_id: str | None = None
    dominant_mood: str | None = None
    avg_arousal: float | None = None
    avg_valence: float | None = None


class ThreadSchema(BaseSchema):
    thread_id: str
    agent_type: AgentType
    title: str | None = None
    brain_context: BrainContext | None = None
    created_at: datetime
    updated_at: datetime


class ThreadCreateSchema(BaseSchema):
    thread_id: str
    agent_type: AgentType


class ThreadSummary(BaseSchema):
    id: str
    thread_id: str
    title: str | None
    brain_context: BrainContext | None = None
    created_at: datetime
    updated_at: datetime


class ThreadListResponse(BaseSchema):
    threads: list[ThreadSummary]


class ThreadMessagesResponse(BaseSchema):
    thread_id: str
    messages: list[dict[str, Any]]
    brain_context: BrainContext | None = None


class ThreadDeleteResponse(BaseSchema):
    message: str


class ThreadRenameRequest(BaseSchema):
    title: str = Field(min_length=1, max_length=255)


class ThreadRenameResponse(BaseSchema):
    thread_id: str
    title: str
