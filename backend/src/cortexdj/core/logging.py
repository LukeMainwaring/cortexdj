import logging
import uuid
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from types import TracebackType

from fastapi import Request
from starlette.datastructures import URL, QueryParams

from cortexdj.core.config import get_settings

# Read once at import on purpose: logging is set up once, as app.py is imported. Do not
# copy this into code a test reconfigures; the Settings seam reaches only code that calls
# get_settings() when it runs.
config = get_settings()

# Query parameters whose values are credentials. A key is one when, lowercased with "-"
# and "_" removed, it contains one of the words below or is one of the exact names
# (`token`, `refresh_token`, `apiKey`, `client-secret`, `password`, `sig`). A browser
# EventSource cannot set headers, so a long-lived stream that needs a token takes it as a
# query parameter; the request line in app.py, uvicorn's access log, the per-line log
# context, and the request spans (core/observability.py) all go through the functions
# below. Only the query string is covered: a JSON body is logged as `request_json`
# unredacted, so a route that takes a credential in its body needs its own handling.
CREDENTIAL_QUERY_WORDS = ("token", "secret", "password", "apikey")
CREDENTIAL_QUERY_NAMES = frozenset({"key", "sig", "signature", "auth", "authorization"})


def _is_credential(key: str) -> bool:
    normalized = key.lower().replace("-", "").replace("_", "")
    return normalized in CREDENTIAL_QUERY_NAMES or any(word in normalized for word in CREDENTIAL_QUERY_WORDS)


def redact_query_params(params: QueryParams) -> QueryParams:
    return QueryParams([(key, "REDACTED" if _is_credential(key) else value) for key, value in params.multi_items()])


def redact_url(url: URL) -> str:
    """The URL as it may be logged: credential query parameters replaced, everything else kept.

    Untouched when there is nothing to redact, so the logged URL is the one the client sent
    (re-encoding the query would turn ``%20`` into ``+``).
    """
    params = QueryParams(url.query)
    if not any(_is_credential(key) for key in params):
        return str(url)
    return str(url.replace(query=str(redact_query_params(params))))


class RedactAccessLogQuery(logging.Filter):
    """Redacts credential query parameters in uvicorn's access log line.

    uvicorn logs each request as ``'%s - "%s %s HTTP/%s" %d'`` with the path and raw query
    as the third arg, and its formatters read the args, so that arg is what gets rewritten.
    Installed on the logger, not a handler, so it holds under any launcher's log config.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and len(record.args) == 5:
            client_addr, method, path, http_version, status_code = record.args
            if isinstance(path, str):
                record.args = (client_addr, method, redact_url(URL(path)), http_version, status_code)
        return True


@dataclass(frozen=True)
class RequestLogContext:
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
                "query_params": redact_query_params(request.query_params),
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

    logging.getLogger("uvicorn.access").addFilter(RedactAccessLogQuery())
    logging.setLoggerClass(ContextualLogger)
    return log_context_var
