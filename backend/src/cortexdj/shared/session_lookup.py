"""The seam other slices read an EEG session's participant through.

Brain-to-track retrieval re-reads a session's raw EEG from the participant's DEAP file, so
it needs the session's participant id, but the sessions slice owns the ``sessions`` table
and a feature may not import another. ``api.py`` registers the sessions slice's lookup here
at import. Only the participant id crosses: nothing else outside the slice needs a session.
"""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager

from sqlalchemy.ext.asyncio import AsyncSession

# (db, session_id) -> the session's participant id, or None when no such session exists.
SessionLookup = Callable[[AsyncSession, str], Awaitable[str | None]]


class SessionLookupNotRegistered(RuntimeError):
    """The session lookup was used before ``cortexdj.api`` installed it.

    A wiring bug in whatever is running (a script, an eval, a test), not a failure the
    model can recover from.
    """


_lookup: SessionLookup | None = None


def register_session_lookup(lookup: SessionLookup) -> None:
    """Install the function that reads a session's participant id; ``api.py`` registers it at import."""
    global _lookup
    _lookup = lookup


@contextmanager
def override_session_lookup(lookup: SessionLookup | None) -> Iterator[None]:
    """Swap the session lookup for the length of a ``with`` block, then put the previous one back.

    For tests: a fake lookup, or ``None`` to stand for "never registered". The app
    registers once at import and never calls this.
    """
    global _lookup
    previous = _lookup
    _lookup = lookup
    try:
        yield
    finally:
        _lookup = previous


async def get_session_participant_id(db: AsyncSession, session_id: str) -> str | None:
    if _lookup is None:
        raise SessionLookupNotRegistered(
            "No session lookup registered: import cortexdj.api (the composition root) first"
        )
    return await _lookup(db, session_id)
