from datetime import datetime

from cortexdj.shared.schema_base import BaseSchema


class NoteSchema(BaseSchema):
    id: str
    title: str
    content: str
    created_at: datetime
    updated_at: datetime


class NoteCreate(BaseSchema):
    title: str
    content: str


class NoteListResponse(BaseSchema):
    notes: list[NoteSchema]
    total: int
