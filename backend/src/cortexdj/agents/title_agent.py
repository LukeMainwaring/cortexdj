"""Thread-title agent: summarizes one exchange into a short conversation title.

Feature-agnostic (any chat-like feature can title a thread), so it lives in the
runtime next to ``build_agent``; the chat feature's ``title_generator.py`` owns
the storage and fallback around it. Obtain it via ``registry.get_agent``.

Deliberately a plain ``Agent`` rather than ``build_agent``: the recipe is for
tool-using agents, and a one-shot title has no tools to guard, no history to
compact, one request to cache, and no need for thinking. Do not route it
through the recipe to "keep things uniform".
"""

from pydantic_ai import Agent

from cortexdj.core.config import get_settings

THREAD_TITLE_AGENT_NAME = "thread_title"

TITLE_INSTRUCTIONS = (
    "Generate a short, descriptive title (3-6 words) for this conversation. "
    "Return ONLY the title text, no quotes or punctuation at the end."
)

# A 3-6 word title needs a few dozen tokens. Capped explicitly because some
# providers (e.g. Anthropic) default max_tokens to the model's full output limit.
TITLE_MAX_TOKENS = 64


def build_title_agent() -> Agent[None, str]:
    settings = get_settings()
    return Agent(
        model=settings.resolve_model(settings.THREAD_TITLE_MODEL),
        name=THREAD_TITLE_AGENT_NAME,
        instructions=TITLE_INSTRUCTIONS,
        model_settings={"max_tokens": TITLE_MAX_TOKENS},
    )
