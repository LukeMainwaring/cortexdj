import logging
import uuid
from contextvars import ContextVar
from types import TracebackType
from typing import Mapping

from fastapi import Request
from pydantic import BaseModel

from cortexdj.core.config import get_settings

config = get_settings()


class RequestLogContext(BaseModel, arbitrary_types_allowed=True):
    request_id: uuid.UUID
    request: Request
    request_json: dict[str, object] | None


log_context_var: ContextVar[RequestLogContext | None] = ContextVar("log_context", default=None)


class ContextualLogger(logging.Logger):
    def _log(
        self,
        level: int,
        msg: object,
        args: tuple[object, ...] | Mapping[str, object],
        exc_info: bool
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | BaseException
        | None = None,
        extra: Mapping[str, object] | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        log_context: RequestLogContext | None = log_context_var.get()

        extra_dict: dict[str, object] = {}
        if extra:
            extra_dict.update(extra)
        if log_context and isinstance(log_context, RequestLogContext):
            request = log_context.request
            extra_dict |= {
                "request_id": log_context.request_id,
                "route": request.url.path,
                "method": request.method,
                "query_params": request.query_params,
                "path_params": request.path_params,
                "client_host": request.client.host if request.client else None,
                "request_json": log_context.request_json,
            }

        super()._log(level, msg, args, exc_info, extra_dict, stack_info, stacklevel)


def setup_logging() -> ContextVar[RequestLogContext | None]:
    if config.ENVIRONMENT == "development":
        logging.basicConfig(
            format="%(asctime)s %(name)s - %(levelname)s:%(message)s",
            level=logging.INFO,
        )

    logging.setLoggerClass(ContextualLogger)
    return log_context_var
