"""Startup and shutdown hook registries.

Lets feature modules register async work to run during application startup
(loading models, warming caches, opening connections, starting background loops)
and to undo it at shutdown, without the lifespan handler in ``app.py`` needing to
import any feature-specific code. The hooks are the observability boot report
(``core/observability.py``) and the model loads ``api.py`` registers: the EEG model
(``features/sessions/eeg_model.py``) and the contrastive encoder
(``features/retrieval/encoder.py``).

To add a hook, call ``register_startup_hook`` (and ``register_shutdown_hook`` for
anything that needs stopping) at import time from a module that ``app.py``
(transitively) imports, e.g. a router module or ``api.py``.
"""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI

StartupHook = Callable[[FastAPI], Awaitable[None]]

_STARTUP_HOOKS: list[StartupHook] = []


def register_startup_hook(hook: StartupHook) -> None:
    """Register a coroutine to run once during application startup."""
    _STARTUP_HOOKS.append(hook)


def get_startup_hooks() -> list[StartupHook]:
    """Return the registered startup hooks in registration order."""
    return list(_STARTUP_HOOKS)


ShutdownHook = Callable[[FastAPI], Awaitable[None]]

_SHUTDOWN_HOOKS: list[ShutdownHook] = []


def register_shutdown_hook(hook: ShutdownHook) -> None:
    """Register a coroutine to run once during application shutdown, e.g. to cancel a task a startup hook began."""
    _SHUTDOWN_HOOKS.append(hook)


def get_shutdown_hooks() -> list[ShutdownHook]:
    """Return the registered shutdown hooks in reverse registration order, so teardown mirrors startup."""
    return list(reversed(_SHUTDOWN_HOOKS))
