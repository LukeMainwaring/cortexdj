# Vercel's chat frontend, October 2026: `vercel/chatbot`, AI Elements, transports, agent UI

Researched 2026-10-15, against primary sources only: GitHub via `gh api` (`vercel/chatbot`, `vercel/ai-elements`, `shadcn-ui/ui`, `shadcn-ui/chatbot-template`), `ui.shadcn.com/docs/ai-elements`, the AI SDK docs bundled with the installed `ai` 7.0.141 (`frontend/node_modules/ai/docs/`), and the installed `pydantic-ai-slim` 2.56.0 source (`backend/.venv/.../pydantic_ai/ui/vercel_ai/`). No blog summaries.
**Confidence note:** the vercel.com changelog and blog were not checked; nothing below depends on them. The AI Elements docs site (`elements.ai-sdk.dev`) was read through its source repo, not the live site, so "the banner is live" is not verified, only that the PR adding it merged. shadcn/ui is maintained by Vercel and is where Vercel's own AI Elements repo now points, so it is treated as first-party here.

## Summary

1. **The direction moved away from both things this frontend descends from.** On 2026-10-14 shadcn/ui released "AI Elements" as copy-in patterns built on new shadcn components (Trace, Code Block, Message Scroller), and `vercel/ai-elements` merged a notice that it "will be deprecated at the end of November 2026" ([PR #519](https://github.com/vercel/ai-elements/pull/519)). The shadcn docs send chatbot builders to a new [`shadcn-ui/chatbot-template`](https://github.com/shadcn-ui/chatbot-template), not to `vercel/chatbot`.
2. **`vercel/chatbot` is quiet and behind us.** Last push 2026-07-08. It moved to AI SDK 7 on 2026-07-06 and pins `ai` 7.0.15 and `@ai-sdk/react` 4.0.16; we are on 7.0.141 and 4.0.145. It is still built on AI SDK Core, Drizzle, NextAuth 5 beta, Redis, and artifacts.
3. **Its chat loop is now layout-mounted.** Since the "v1" rewrite (2026-03-20), one `useChat` lives in `ActiveChatProvider` in `app/(chat)/layout.tsx`, and both chat pages render `null`. Two features look present but are switched off: the resume endpoint returns 204 unconditionally (since 2026-01-15), and no tool requires approval (v1 removed `needsApproval` from `getWeather`), though the approval UI is still rendered.
4. **Transports: nothing new we need.** The installed docs describe `DefaultChatTransport`, `TextStreamChatTransport`, `WebSocketChatTransport`, `DirectChatTransport`, and `WorkflowChatTransport` (from `@ai-sdk/workflow`). Ours is `DefaultChatTransport` with `prepareSendMessagesRequest` sending only the last message, which is the shape the docs' own resumable-stream example uses.
5. **The agent UI contract is richer than what we render, and our backend already speaks most of it.** AI SDK 7 tool parts carry `approval-requested`, `approval-responded`, and `output-denied` states; `toolApproval` replaces `needsApproval`; automatic approvals arrive in the same states with `isAutomatic: true`. Pydantic AI 2.56's `VercelAIAdapter` emits `tool-approval-request` chunks for deferred approvals and reads approval responses back. It also has a `DataChunk` for `data-*` parts.
6. **Worth adopting: little, and nothing structural.** There is one optional UI swap (shadcn Message Scroller). Two agent-UI patterns (tool approval, a structured questionnaire) are worth adopting only if a product reason appears. Do not adopt the template's chat loop, Core route, resumable streams, Drizzle, or auth. Details are in section 5.

## 1. `vercel/chatbot` today

**Repo state.** `vercel/chatbot`, about 21k stars, not archived, `pushed_at` 2026-07-08 (`gh api repos/vercel/chatbot`). The tree has 181 blobs (`git/trees/main?recursive=1`). Top-level shape: `app/(auth)/` (NextAuth routes, login, register), `app/(chat)/api/{chat,chat/[id]/stream,document,files/upload,history,messages,models,suggestions,vote}`, `artifacts/{code,image,sheet,text}`, `components/ai-elements/` (9 files), `components/chat/` (44 files), `hooks/` (`use-active-chat.tsx`, `use-artifact.ts`, `use-auto-resume.ts`, `use-chat-visibility.ts`, `use-messages.tsx`, `use-mobile.ts`, `use-scroll-to-bottom.tsx`), `lib/ai/` (models, prompts, providers, 5 tools), `lib/db/` (Drizzle schema, one squashed migration), `lib/editor/` (ProseMirror), `tests/e2e/` (Playwright).

**Versions, theirs vs ours** (`package.json` on `main` vs `frontend/package.json`):

| Package | `vercel/chatbot` | ours |
|---|---|---|
| `ai` | `7.0.15` (exact) | `^7.0.141` |
| `@ai-sdk/react` | `4.0.16` (exact) | `^4.0.145` |
| `next` | `16.2.10` | `16.4.1` |
| `react` / `react-dom` | `19.2.7` | `^19.3.1` |
| `streamdown` | `^2.3.0` (+ `@streamdown/{code,math,mermaid,cjk}`) | `^2.7.0` |
| `lucide-react` | `^0.446.0` | `^1.57.0` |
| `typescript` | `^5.6.3` | TypeScript 6 alias |
| `ultracite` / Biome | `^7.8.4` / `2.5.2` | `8.0.1` / `2.5.15` |
| Core-side deps | `drizzle-orm`, `postgres`, `next-auth` 5 beta, `redis`, `resumable-stream`, `@vercel/blob`, `botid`, `swr`, `@ai-sdk/otel` | none |

It also carries `babel-plugin-react-compiler` and `"style": "radix-maia"` in `components.json`; ours is `"style": "default"`.

**Chat-loop structure.**
- `app/(chat)/layout.tsx` mounts `DataStreamProvider` → `SidebarProvider` → `ActiveChatProvider` → `ChatShell`, then `{children}`. `app/(chat)/page.tsx` and `app/(chat)/chat/[id]/page.tsx` are both `export default function Page() { return null; }`. The PR body for v1 ([#1462](https://github.com/vercel/chatbot/pull/1462)) states the goal: "persistent chat shell that never unmounts between chat switches".
- `hooks/use-active-chat.tsx` derives `chatId` from `usePathname()` (a fresh UUID for `/`), loads history with SWR from `/api/messages?chatId=`, and calls `setMessages` when it arrives. It owns `useChat` with:
  - `onData` routing `data-waiting-status` to a waiting indicator and every other data part into `DataStreamProvider` (artifacts, title).
  - `sendAutomaticallyWhen` set to an inline check for an approved `approval-responded` part.
  - A `DefaultChatTransport` whose `prepareSendMessagesRequest` sends `{ message: lastMessage }` normally and the full `messages` only for a tool-approval continuation, plus `selectedChatModel` and `selectedVisibilityType`.
- `useAutoResume` calls `resumeStream()` when the last loaded message is a user message. On the server, `consumeSseStream` writes the SSE stream to Redis when `REDIS_URL` is set. But `app/(chat)/api/chat/[id]/stream/route.ts` is three lines returning `new Response(null, { status: 204 })`, stubbed in [#1392](https://github.com/vercel/chatbot/pull/1392) (2026-01-15, no PR description). In practice resume never replays anything. Why it was stubbed: not verified.
- `components/chat/message.tsx` still renders `ToolApprovalActions` (Allow/Deny calling `addToolApprovalResponse`). However, no tool has approval configured: v1 deleted `needsApproval: true` from `lib/ai/tools/get-weather.ts`, and `app/(chat)/api/chat/route.ts` passes no `toolApproval` to `streamText`.
- `app/(chat)/api/chat/route.ts` (470 lines) runs `streamText` inside `createUIMessageStream`, with `stopWhen: isStepCount(5)`. It writes transient `data-waiting-status` parts ("Waiting...", then "Still waiting..." or a model-health warning after 9 s) and `data-chat-title`. Around that sit BotID, IP rate limiting, and entitlements.

**Commit themes** (`commits?per_page=50`, 2025-12-12 to 2026-07-08):
- **Dec 2025:** AI SDK 6 beta plus a tool-approval demo ([#1361](https://github.com/vercel/chatbot/pull/1361)), Playwright e2e, dynamic model discovery from AI Gateway, token-usage tracking removed.
- **Feb–Mar 2026:** a rebrand to "openchat" and back, base-path work, BotID and IP rate limiting, NextAuth to better-auth ([#1453](https://github.com/vercel/chatbot/pull/1453)) and then back again in v1 ("removed better-auth"), a sidebar redesign, and the v1 persistent shell.
- **Apr–May:** model list churn.
- **July:** AI SDK 7 ([#1515](https://github.com/vercel/chatbot/pull/1515), 9 files, mostly renames), Zod 4, lint tooling, model-health warning, pre-commit hook.
- **Since 2026-07-08:** nothing.

The template's recent work is about operating a public demo (gateway, models, abuse, auth). None of it changes the AI SDK UI contract.

## 2. AI Elements

**What it was.** `vercel/ai-elements` describes itself as "a component library and custom registry built on top of shadcn/ui". It ships 49 components in `packages/elements/src/`, among them `message`, `conversation`, `prompt-input`, `tool`, `confirmation`, `reasoning`, `chain-of-thought`, `sources`, `inline-citation`, `plan`, `queue`, `task`, `shimmer`, `suggestion`, `code-block`, `question` (added 2026-08-21), and a React Flow set. **Distribution is a shadcn registry:** `npx shadcn@latest add https://elements.ai-sdk.dev/api/registry/<name>.json`. The `ai-elements` CLI (v1.9.1) is a thin wrapper that builds those URLs and runs `shadcn add` (repo `README.md`, `packages/cli/index.js`). Code is copied into the app, which is why the template has its own `components/ai-elements/`. Its `message.tsx` renders markdown with `Streamdown`, `conversation.tsx` wraps `use-stick-to-bottom`, and `tool.tsx` and `confirmation.tsx` already handle the approval states.

**What replaces it.** The [shadcn changelog of 2026-10-14](https://github.com/shadcn-ui/ui/blob/main/apps/v4/content/docs/changelog/2026-10-ai-elements.mdx) releases Trace ("a list of lines for the steps an agent takes") and Code Block. It also releases [AI Elements](https://ui.shadcn.com/docs/ai-elements): "There is no package to install. Each pattern is a composition you copy into your project". There are 21 patterns, among them Chat, Message, Prompt Input, Tool, Confirmation, Reasoning, Sources, Plan, Queue, Subagent, and Suggestion. Each page shows rendering from AI SDK message parts and is available as `.md` "so you can point your coding agent at it". The building blocks are the June 2026 chat components (Message Scroller, Message, Bubble, Marker, Attachment, Questionnaire) plus Trace and Code Block. They come for Base UI, React Aria and Radix. Message Scroller needs `@shadcn/react`. On migration: "You do not need to migrate right away. The components you installed from AI SDK Elements were copied into your project, so they keep working." Each pattern page has a prompt that rebuilds the wrapper and keeps its exports and props. Twenty-three elements (artifact, the React Flow canvas set, web-preview, persona, and others) were not rebuilt.

**The reference chatbot changed too.** [`shadcn-ui/chatbot-template`](https://github.com/shadcn-ui/chatbot-template) was created 2026-08-11 and last pushed 2026-10-15 ("move chat UI to Trace and AI Elements"). It is one page with no persistence and no auth. It runs `useChat` with the default transport and a Core route (`streamText`, `validateUIMessages`, `isStepCount(5)`). It uses `react-markdown` rather than Streamdown, `@base-ui/react`, and `"style": "base-rhea"`. Its one HITL feature is an `ask_user` tool with no `execute`, answered in a Questionnaire through `addToolOutput` and continued with `sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls` (`tools/ask_user.ts`, `components/chat.tsx`). The installed AI SDK docs do not mention AI Elements at all (`grep -r "AI Elements\|ai-elements" frontend/node_modules/ai/docs` is empty).

**Overlap with `frontend/components/elements/`** (324 lines across six files):

| ours | `vercel/ai-elements` | shadcn AI Elements |
|---|---|---|
| `message.tsx` (`MessageContent`, 20 lines) | `message.tsx` (`Message`, `MessageContent`, actions, branches) | Message pattern, Bubble |
| `response.tsx` (memoised `Streamdown`) | `MessageResponse` in `message.tsx` | none; template uses `react-markdown` + shadcn/typeset |
| `actions.tsx` (`Actions`, `Action` with tooltip) | `MessageActions`, `MessageAction` | Message footer actions |
| `tool-call.tsx` (collapsible, verbs, hides output when a panel renders) | `tool.tsx` | Tool pattern on Trace + Code Block |
| `prompt-input.tsx` (106 lines) | `prompt-input.tsx` (1,341 lines in the template's copy) | Prompt Input pattern |
| `bouncing-dots.tsx` + `animate-shimmer` in `message.tsx` | `shimmer.tsx` | `shimmer` util, Trace running state |
| `components/messages.tsx` + `hooks/use-scroll-to-bottom.tsx` | `conversation.tsx` (`use-stick-to-bottom`) | Chat pattern on Message Scroller |

Our `tool-call.tsx` has no approval or denied states, since nothing asks for approval. No equivalent exists anywhere for the tool-panel registry (`features/tool-panel-registry.ts`). Upstream renders each tool by a `switch` on `part.type` in `message.tsx`, in both templates.

## 3. Transports

From `frontend/node_modules/ai/docs/04-ai-sdk-ui/21-transport.mdx`, `02-chatbot.mdx`, and `03-chatbot-resume-streams.mdx`:

- **`DefaultChatTransport`**: HTTP POST to `/api/chat`. `headers`, `body`, and `credentials` can be functions. `prepareSendMessagesRequest` reshapes the body, and `prepareReconnectToStreamRequest` sets the resume URL (default `GET /api/chat/[id]/stream`).
- **`TextStreamChatTransport`**: for plain text streams. "tool calls, usage information and finish reasons are not available". Not useful to us.
- **`WebSocketChatTransport`**: one socket shared across turns, multiplexed by `requestId`. It "does not automatically replay requests", and `reconnectToStream` asks for "a complete replay ... from sequence zero". The server must implement its wire protocol.
- **`DirectChatTransport`**: calls an in-process `Agent.stream()`. It is for SSR, tests, and desktop apps, and needs an AI SDK `Agent`.
- **`WorkflowChatTransport`** (`@ai-sdk/workflow`): reconnects when a workflow function times out mid-stream.
- **Custom**: implement the `ChatTransport` interface; the docs point at the `DefaultChatTransport` source.
- **Resumable streams**: `useChat({ resume: true })` GETs the resume endpoint on mount and "when the page becomes visible again" after a network drop. The server side needs `resumable-stream`, Redis, an `activeStreamId` per chat, and a separate stop endpoint, because "client-side aborts are treated as disconnects". Replays are full by default (`resumeStreamIsReplay`).

**What the templates do.** `vercel/chatbot` uses `DefaultChatTransport` with a custom `fetch` and a last-message body. Its resume endpoint is stubbed (section 1). `shadcn-ui/chatbot-template` uses the default transport with no options.

**Ours.** `frontend/components/chat.tsx` builds `DefaultChatTransport({ api: "/api/chat", prepareSendMessagesRequest })` sending `{ id, trigger, messages: messages.slice(-1) }`, and `route.ts` proxies the body to FastAPI. This is already the shape the docs recommend for server-stored history. Resume would need a stream store and a stop endpoint in FastAPI, and Stop would no longer cancel the run. That reverses the turn semantics in `docs/architecture/one-chat-turn.md`.

## 4. Agent patterns that reach the UI

- **Tool approval** (`04-ai-sdk-ui/03-chatbot-tool-usage.mdx`, `03-agents/06-tool-approvals.mdx`):
  - Typed tool parts gain `approval-requested`, `approval-responded`, and `output-denied`, and the docs say to "include these states when handling `part.state` exhaustively, even when a tool does not require approval".
  - Server rules return `'not-applicable' | 'approved' | 'denied' | 'user-approval'`. Automatic decisions arrive with `part.approval.isAutomatic === true` and an optional `reason`.
  - The client answers with `addToolApprovalResponse({ id, approved, reason? })` and continues with `sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses`. `experimental_toolApprovalSecret` signs approvals, because a modified client could fabricate them.
  - **Backend side:** Pydantic AI 2.56 `VercelAIEventStream` yields `ToolApprovalRequestChunk` for each `DeferredToolRequests.approvals` entry when `sdk_version >= 6` (`_event_stream.py`). `VercelAIAdapter.deferred_tool_results` turns `approval-responded` parts back into `DeferredToolResults` (`_adapter.py`). Whether that works with our last-message-only body and server-stored history in `features/chat/stream.py` is not verified.
- **Client-side tools / questionnaires**: a tool with no `execute` shows `input-available`. The UI answers with `addToolOutput` and auto-continues with `lastAssistantMessageIsCompleteWithToolCalls`. This is the `ask_user` pattern in the shadcn template. On our side the adapter builds `DeferredToolResults` from approvals only. A client-supplied output comes back as an ordinary tool part in the request, and attaching it to the stored history would be our code (not verified).
- **Data parts** (`04-ai-sdk-ui/20-streaming-data.mdx`): `data-*` parts are persistent (in `message.parts`), transient (only via `onData`), or reconciled by `id` ("write to a data part with the same ID" to update it). Pydantic AI has `DataChunk(type='data-…', id, data, transient)` in `response_types.py`. Our `ChatMessage` declares no data parts (`frontend/lib/types.ts`).
- **Subagents** (`03-agents/06-subagents.mdx`): streamed subagent progress arrives as `output-available` with `preliminary: true` until the final output. No tool approvals inside subagents. A delegation to a specialist agent renders as one tool call. Preliminary outputs would need the backend to stream them, which Pydantic AI's adapter does not do as far as checked (not verified).
- **Agents and harnesses**: AI SDK 7 has `ToolLoopAgent`, `WorkflowAgent`, and `HarnessAgent`. `03-ai-sdk-harnesses/07-ui.mdx` says harness streams are UI-message streams, and that the route should resume a `HarnessAgentSession` "instead of replaying the whole UI message history". This is the same server-owns-history stance this project takes, but all of it is AI SDK Core on the server. For a `useChat` client the only visible effect is the part types above.
- **Reasoning**: `sendReasoning` forwards `reasoning` parts, and Pydantic AI emits `reasoning-*` chunks. `components/message.tsx` renders only `text` and tool parts, so reasoning parts are dropped if thinking is on. An instance that turns thinking on decides whether to render them.

## 5. Worth adopting?

**Candidates**
- **shadcn Message Scroller in place of `hooks/use-scroll-to-bottom.tsx`.** It anchors a new turn near the top, does not pull a reader who has scrolled away, and reopens a saved thread at the last user message. Size: small to medium, meaning one shadcn component plus `@shadcn/react`, and a rewrite of `components/messages.tsx`. Worth it only if the current scroll behaviour bothers the user in live checks. Whether our `"style": "default"` `components.json` can pull the new components without switching styles is not verified.
- **Tool approval on a destructive tool, with a Confirmation-style row.** The backend adapter already emits and reads approval parts. The cost is `requires_approval` on the tool, `DeferredToolRequests` in the assistant's output types, approval states in `tool-call.tsx`, `addToolApprovalResponse` wiring in `chat.tsx`, and a check that `stream.py` handles a continuation whose last message is the assistant's. Size: medium. Adopt only if there is a product reason to make that action a two-step one.
- **A structured questionnaire (client-answered tool plus a Questionnaire, as in `ask_user`).** Size: medium to large, because the result of a client-answered tool has to be stitched onto server-stored history, which is new backend code. It would be its own slice-level piece of work, and only with a product reason.

**Not worth adopting**
- **The layout-mounted chat (`use-active-chat.tsx`).** It solves chat switches without unmounting, in an app with SWR, model selection, visibility, and votes. We have none of those, and `chat.tsx` is keyed per thread with TanStack Query owning history.
- **The template's transport shaping** (`message` vs full `messages` for approval continuations, model and visibility in the body). It exists for Core's `convertToModelMessages`; our backend owns history.
- **Resumable streams.** They need Redis, an active-stream record, and a stop endpoint, and Stop would no longer stop the run. The template itself has the resume route stubbed.
- **AI SDK Core in `route.ts`, `DirectChatTransport`, `ToolLoopAgent`, harnesses, `WorkflowChatTransport`.** All of these assume the agent runs in Node. Ours runs in Pydantic AI, in the FastAPI backend.
- **Drizzle, NextAuth, BotID, rate limiting, AI Gateway model discovery, artifacts.** These are public-demo and multi-user concerns that this template replaces with FastAPI.
- **A wholesale move to AI Elements, either version.** The Vercel set is being deprecated. The shadcn set is copy-in patterns, so there is nothing to "adopt" beyond individual files. Our six `elements/` files cover what we render, and the tool-panel registry has no upstream equivalent to converge on.
- **`WebSocketChatTransport`.** It needs a new server protocol, and per its docs it only helps "chats with frequent client-side tool results".
