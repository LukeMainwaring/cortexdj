---
paths:
  - "frontend/components/**/*.tsx"
  - "frontend/app/(chat)/**/*.{ts,tsx}"
  - "frontend/hooks/**/*.{ts,tsx}"
  - "frontend/api/hooks/**/*.{ts,tsx}"
  - "frontend/lib/**/*.ts"
---

# Vercel AI SDK rules

## Docs ship inside the installed package

The `ai` package bundles its full documentation as MDX, matched to the installed
version. Grep it before writing any AI SDK code: `useChat` and the UI APIs
change often across versions, so remembered APIs are usually stale.

1. **`frontend/node_modules/ai/docs/`**: the whole doc set. UI guides in
   `04-ai-sdk-ui/` (chatbot, transports, the SSE stream protocol, tool usage);
   API reference in `07-reference/02-ai-sdk-ui/` (`useChat` parameters and
   return values live in the `<PropertiesTable>` blocks). Source is alongside
   in `node_modules/ai/src/`.
2. **`https://ai-sdk.dev/docs`**: WebFetch fallback for providers, cookbook,
   and anything the bundled docs lack. Append `.md` to a page URL for markdown.

## AI SDK UI only

`app/(chat)/api/chat/route.ts` is a thin proxy to the FastAPI backend's `POST /agent/chat`;
Pydantic AI emits the SSE stream. When the docs show `streamText` /
`toUIMessageStreamResponse`, that's a **spec for what the backend must emit**, not code to
write here.

## Where to copy built UI from

The docs above specify the contract and contain no production components. For
built chat UI (attachments, message actions, tool rows, scroll), copy from
shadcn's AI Elements patterns (`https://ui.shadcn.com/docs/ai-elements`, each
page fetchable as `.md`) or `shadcn-ui/chatbot-template`, once the docs have
settled the contract. Lift the markup, Tailwind, and interaction states; leave
their wiring: Core `streamText` routes, Drizzle, auth, a layout-mounted chat
context, their transport setup. This frontend was forked from `vercel/chatbot`
but is ahead of it and divergent by design, so it is not an upstream to resync
with, and `vercel/ai-elements` is being deprecated. Background:
`docs/research/vercel-chatbot-direction.md`.
