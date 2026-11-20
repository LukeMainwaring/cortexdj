"""Feature-slice registry for Alembic metadata discovery.

Importing this package imports every feature's model module, so each table is
registered on ``Base.metadata`` before Alembic autogenerate runs (see
``migrations/env.py``). Import models ONLY here — never routers or services —
to keep this side-effect-free beyond table registration.
"""

from cortexdj.features.chat.model import Message, Thread
from cortexdj.features.sessions.model import EegSegment, Session, SessionTrack, ThreadBrainContext, Track

__all__ = [
    "EegSegment",
    "Message",
    "Session",
    "SessionTrack",
    "Thread",
    "ThreadBrainContext",
    "Track",
]
