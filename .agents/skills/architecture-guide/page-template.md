# Page template

Each section stays within about one screen: a page is something to study.

## Topic page

File name is the topic as a slug, for example `one-chat-turn.md`.

````markdown
# <Topic>

## Summary

Two or three sentences: what this is and the one idea to remember.

## Diagram

```mermaid
<one diagram>
```

## How it works

Short prose on the main pieces and how they connect. Name files and functions
so the reader can find them; leave the line-by-line to the code.

## Why it's built this way

The trade-offs behind the shape, each linked to the ADR, `AGENTS.md` section,
or docstring that states it.

## How to explain this

**In about a minute:**

> First person, the way you would say it out loud to another engineer. Plain
> sentences, one idea each, in the order you would draw it on a whiteboard.

**Likely follow-ups:**

- **<A question an interviewer would plausibly ask>** Two-sentence answer.
- three to five of these

## Sources

Written at commit `<short sha>`.

- `<path or directory>`
````

### Diagram

When the topic has a rendered diagram, one line after the Mermaid block links it: `Rendered with archify: [diagrams/<slug>](diagrams/<slug>/<slug>.html) (open from a local checkout).`

Pick the Mermaid type from the topic: `sequenceDiagram` for a request's path, `flowchart` for structure or data movement, `stateDiagram-v2` for phases. Keep it to what the summary claims, around fifteen nodes at most, with a verb on every arrow. A detail that needs a second diagram belongs in its own topic.

### How to explain this

The spoken version starts from what the system does for its user and reaches the mechanism second. Follow-ups are the questions that probe a decision ("why not let the client choose the agent?", "what happens if the run fails halfway?"), and each answer states the decision and its reason.

### Sources

List the code and ADRs the page's claims rest on, repo-relative. Prefer a directory when the topic covers everything in it, so new files there mark the page stale. Cite a broad document such as `AGENTS.md` in the prose only: it changes for unrelated reasons and would mark the page stale each time.

## Overview page

`docs/architecture/README.md` uses the same sections with three differences:

- After the summary, a **Topics** section: one line per topic, linking its page and saying what question it answers. This list is the topic list the skill reads.
- It states only what its own sources or a topic page support, and leaves counts and other incidental details to the topic pages. Its sources are the few files that change when the system's shape changes (the composition roots, the routing function), so ordinary feature work leaves the overview current.
- After the sources, a **Watched for new topics** list: the directories where a new entry means a new slice, agent, or decision (for example the features directory and the ADR directory).
