---
paths:
  - "frontend/components/**/*.tsx"
  - "frontend/app/(chat)/**/*.{ts,tsx}"
  - "frontend/hooks/**/*.{ts,tsx}"
  - "frontend/api/hooks/**/*.{ts,tsx}"
  - "frontend/lib/**/*.ts"
---

# Vercel AI SDK Rules

## Docs are split between two places

The Vercel AI SDK's documentation lives in two places with **different content**:

1. **`docs/vercel-ai-sdk-ui.txt`** — local pinned reference, **UI surface only**
   (`useChat`, message parts, chat transports, the SSE stream protocol,
   tool-call rendering). Refresh via the `updating-deps` skill (Phase 4 is the
   source of truth for the fetch).

2. **`https://ai-sdk.dev/`** — everything else. AI SDK Core (`generateText`,
   server-side `streamText`, model providers), guides, framework examples,
   cookbook. Not cached locally; fetch ad-hoc with WebFetch.

**Rule of thumb:** if you're grepping `vercel-ai-sdk-ui.txt` for anything
outside the UI slice and finding nothing, it hasn't been deleted — it's on
the web docs. Use WebFetch on `https://ai-sdk.dev/docs/...` before giving up.

## The chat surface in this project

- **AI SDK UI only — no Core server-side.** `app/(chat)/api/chat/route.ts` is
  a thin proxy to the FastAPI backend's `POST /agent/chat`; Pydantic AI emits
  the SSE stream. When the docs show `streamText` / `toUIMessageStreamResponse`,
  that's a **spec for what the backend must emit**, not code to write here.
- **Keep `ChatMessage` threaded.** `useChat` is parameterized by the custom
  `ChatMessage` type in `frontend/lib/types.ts`. Carry that generic through
  every `UseChatHelpers` site — falling back to plain `UIMessage` loses the
  project's custom data/tool-part typing.
- **Tool-call panels switch on `part.type === "tool-<name>"`** in the message
  renderer. A new backend tool that needs a custom UI panel adds a branch
  there.
