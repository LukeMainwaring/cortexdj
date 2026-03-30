"""Health check response schema."""

from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str
