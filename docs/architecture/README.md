# Architecture guide

## Summary

CortexDJ is a chat app where a user talks to one assistant that can call tools over the app's own data; the shipped domain is Notes. A Next.js frontend proxies each chat turn to a FastAPI backend, which stores the turn, runs a Pydantic AI agent against the thread's history, and streams the reply back as server-sent events. The one idea to remember: the backend is cut into feature slices that only meet at a few composition roots, and an instance grows by copying a slice.

## Topics

- [Slices and the dependency rule](slices-and-the-dependency-rule.md): how is the code organized, and what is allowed to import what?
- [One chat turn](one-chat-turn.md): what happens between the browser sending a message and the reply finishing?
- [The evaluation loop](the-evaluation-loop.md): how do you know the agent is getting better, and what does a run leave behind to look at?

## Diagram

```mermaid
flowchart LR
    browser["Browser<br/>useChat"] -->|sends newest message| proxy["Next.js<br/>/api/chat proxy"]
    proxy -->|forwards, pipes SSE back| stream["Chat stream<br/>POST /api/agent/chat"]
    stream -->|reads and appends history| threads[("threads + messages")]
    stream -->|runs| assistant[Assistant agent]
    assistant -->|list_notes, create_note, search_notes| notes[("notes<br/>pgvector + tsvector")]
    assistant -->|calls| model[Model provider]
    notes -.->|embeds on create and search| embedder[Embedding model]
    stream -.->|titles the thread after the first reply| title[Title agent]
    title -->|calls| model
```

## How it works

The backend package is `cortexdj`. A thin kernel (`core/`, `shared/`, `agents/`) holds what every feature shares, and each domain is a slice under `features/`: `chat` (threads, messages, title generation, and the streaming endpoint) and `notes` (the example domain: a table, routes, and the assistant's three tools, with the hybrid search, a pgvector semantic leg and a full-text keyword leg, marked as the cuttable `embeddings` feature).

A chat turn enters at `features/chat/stream.py`. It stores the user's message, loads the thread's history, gets the assistant from the registry in `agents/registry.py`, and streams the run back in the Vercel AI SDK format. The assistant is built once by `build_assistant` in `agents/assistant.py` from `ENABLED_CAPABILITIES`, the list in `agents/capabilities.py`; the notes slice's capability is the one entry. Threads, messages, and notes live in one Postgres database, and every route takes a session from `core/db.py` that commits when the request ends.

The frontend is a Next.js chat shell. `components/chat.tsx` sends only the newest message, `app/(chat)/api/chat/route.ts` forwards it, and `components/message.tsx` renders tool calls, mounting a richer panel when the tool's name is in `features/tool-panel-registry.ts`.

## Why it's built this way

`AGENTS.md` describes the template's purpose: a vertical slice (`notes`) to copy when adding a domain, and small registries so a new router, capability, agent, or startup hook is one registration rather than edits scattered across `app.py`. The `.template/` markers let onboarding prune the agent, the frontend, embeddings, or scrubbing and leave a working app.

The decisions that shaped specific parts are recorded as ADRs: how tests are tiered and where they may fake ([0001](../adr/0001-test-tiers-and-fakes.md)), why prompt caching is part of the shared recipe with each prompt split into static rules and per-run context ([0002](../adr/0002-prompt-caching-in-the-shared-recipe.md)), why the facts a trace is searched by ride on the agent run as metadata and nothing else is traced by hand ([0003](../adr/0003-run-metadata-not-a-custom-chat-turn-span.md)), and why the eval judges run on a model a tier above the agents, pinned apart from them ([0004](../adr/0004-eval-judges-a-tier-above-the-agents.md)).

## How to explain this

**In about a minute:**

> It's a chat app template. The user talks to one assistant, built with Pydantic AI, that has tools over the app's own data; the shipped example is notes it can list, create, and search by meaning. The frontend is a thin Next.js chat UI that sends only the newest message; all the logic is in a FastAPI backend, which owns the conversation history, saves the user's turn before the model runs, and streams the reply back as server-sent events. The backend is organized as feature slices, each with its own model, service, routes, and agent tools, over a small kernel. Slices never import each other; they meet in three small files that wire everything together. To add a domain you copy the notes slice on each side. Around that sits the part that makes it a template: tests in three tiers, an eval tier that measures the agent as it ships, tracing through the standard instrumentation, and markers that let you cut the features you don't want when you start.

**Likely follow-ups:**

- **Where does state live?** In Postgres: the thread history per thread, and the notes. The browser holds nothing the server needs, so it sends only its newest message, and a regenerate reuses the stored turn.
- **How would you add a domain?** Copy `features/notes/` on each side: model, schema, service, router, then tools and a capability, and register the router in `api.py` and the capability in `agents/capabilities.py`. The frontend side is a panel and one line in the tool-panel registry.
- **How would you add another agent?** A builder that calls `build_agent` with its capabilities, one line in `agents/registry.py`, and, if it needs its own model, a setting resolved with `Settings.resolve_model`.
- **What is the retrieval piece?** Hybrid search over notes: a semantic leg (pgvector with an HNSW index, an embedding written best-effort when a note is created) and a keyword leg (a generated `tsvector` with a GIN index), fused by reciprocal rank in Python with no weights to tune, and each result says which leg found it ([ADR 0005](../adr/0005-hybrid-notes-search-fused-by-reciprocal-rank.md)). It is marked as a cuttable feature so an instance without it still builds.

## Sources

Written at commit `039bd25`.

- `backend/src/cortexdj/api.py`
- `backend/src/cortexdj/agents/capabilities.py`
- `backend/src/cortexdj/agents/registry.py`
- `backend/src/cortexdj/features/__init__.py`
- `frontend/features/tool-panel-registry.ts`

Watched for new topics:

- `backend/src/cortexdj/features/`
- `backend/src/cortexdj/agents/`
- `frontend/features/`
- `docs/adr/`
