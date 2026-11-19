from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class AgentDeps:
    """Per-run dependencies, reachable from tools as ``ctx.deps``.

    The thread id is deliberately not here: the chat endpoint passes it as the
    run's ``conversation_id``, so tools read it from ``ctx.conversation_id``.
    """

    db: AsyncSession
