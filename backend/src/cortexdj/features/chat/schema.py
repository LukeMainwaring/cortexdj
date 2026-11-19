from datetime import datetime
from typing import Any

from pydantic import Field

from cortexdj.shared.schema_base import BaseSchema


class ThreadSummary(BaseSchema):
    thread_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class ThreadListResponse(BaseSchema):
    threads: list[ThreadSummary]


class ThreadMessagesResponse(BaseSchema):
    thread_id: str
    messages: list[dict[str, Any]]


class ThreadDeleteResponse(BaseSchema):
    message: str


class ThreadRenameRequest(BaseSchema):
    title: str = Field(min_length=1, max_length=255)


class ThreadRenameResponse(BaseSchema):
    thread_id: str
    title: str
