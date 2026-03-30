"""Base schema with ORM mode for all ORM-mapped schemas."""

import pydantic


class BaseSchema(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(from_attributes=True)
