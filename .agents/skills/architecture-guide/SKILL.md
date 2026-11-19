---
name: architecture-guide
description: Write or update the architecture guide under docs/architecture, one page per topic, and optionally render a topic with archify.
argument-hint: "[check | full | render <topic>]"
disable-model-invocation: true
---

The **architecture guide** lives in `docs/architecture/`: `README.md` is the overview and holds the topic list, and each **topic** (one way of looking at the system, such as "one chat turn") is one page beside it. The pages are the overview and the topics it lists; `diagrams/` holds rendered diagrams and is not a page. The reader is an engineer learning the system well enough to explain it to someone else. Every page follows [`page-template.md`](page-template.md).

Read a page's source files before writing or changing it, and write only what they show. Reasons come from ADRs, `AGENTS.md`, and docstrings; where none is written down, the page says the reason is unrecorded.

## Pick the run

- No `docs/architecture/README.md`: **first run**.
- Argument `check`: **find stale pages** (and stale diagrams), **find new entries**, report both, and stop.
- Argument `full`: rewrite every page in the topic list from its sources, **find new entries**, then **report**.
- Argument `render <topic>`: re-render that topic's diagram by following [`archify.md`](archify.md), then stop. Nothing else is read or changed.
- Otherwise: **find stale pages**, **update** them, **find new entries**, then **report**.

The guide describes committed code: every comparison runs from a page's recorded commit to `HEAD`. When the working tree has uncommitted changes under a page's sources, say in the report that they will be picked up once committed.

## First run

1. Read `AGENTS.md`, the ADRs, and the directory layout, then propose a topic list: the overview plus the few topics that carry the design. Favor topics you would explain out loud (a request's path, a rule the layout enforces, where data comes from) over one topic per module.
2. Wait for the user to confirm or change the list.
3. Write the overview and each topic page. Done when every confirmed topic has a page with all template sections and a sources footer.

## Find stale pages

Each page's footer records the commit it was written at and its source paths. For every page, overview included:

```bash
git diff --name-only <commit> HEAD -- <source paths>
```

A page is **stale** when that prints any file, or when the commit no longer exists.

A page whose Diagram section has a "Rendered with archify" link also has a rendered diagram, and its `candidate.json` records the commit it was drawn at in `meta.repository.revision`. Run the same diff from that commit over the page's source paths. The **diagram is stale** when it prints any file, even if the page is current, so a page updated earlier without a re-render is still caught.

Done when every page and every rendered diagram is marked stale or current, with the changed files listed for each stale one.

## Update

For each stale page:

1. Read the diff of its changed files and the page.
2. Edit the sentences, diagram nodes, and talking points the change touches. Leave the rest of the page exactly as it is, so the page's own diff shows what changed in the system.
3. Set the footer to `HEAD`'s commit, and add or drop source paths if the topic's files moved. A change that leaves the page's story intact still gets the new commit. The overview is the exception: edit its text here, but its footer is set at the end of **find new entries**, since that step uses the old commit.

When any topic page changed, re-read the overview and fix whatever no longer matches it.

## Find new entries

The overview's footer has a "Watched for new topics" list of directories. List what was added there since the overview's commit:

```bash
git diff --name-only --diff-filter=A <overview commit> HEAD -- <watched directories>
```

For each addition (a new slice, a new agent, a new ADR), read enough to judge whether an existing page already explains it. A page listing the directory as a source is not the same as explaining what was added. Report the ones no page explains as possible new topics, or as additions to an existing page.

Outside `check`, wait for the user's decision on each, then set the overview's footer to `HEAD`'s commit so settled entries are not raised again. Adding or removing a page is the user's call.

## Report

List each page touched, the changed files that triggered it, and one line on what you edited. List current pages as a count, then any new entries and what was decided. List each stale diagram with the files that made it stale. Then offer to render: each stale diagram, and any topic with no rendering yet. For each the user accepts, follow [`archify.md`](archify.md). A declined stale diagram stays stale and is listed again next run; `render <topic>` picks it up later.
