# Rendering a topic with archify

Archify is a separately installed skill that draws a **rendered diagram** as standalone HTML. The guide is complete without it: if the `archify` skill is not available, say so and stop here.

For each topic the user picks:

1. **Choose the type** from the topic: `sequence` for a request's path, `dataflow` for where data comes from and goes, `lifecycle` for phases and states, `architecture` for structure.
2. **Use one folder per topic**, `docs/architecture/diagrams/<topic-slug>/`, and tell archify to use it, with the HTML named `<topic-slug>.html`. Keep only the spec and the HTML there; the receipts rule in step 4 says when to delete the rest.

   If the folder already has a `candidate.json`, this is a **re-render**. First rename `<topic-slug>.html` to `<topic-slug>-previous.html` in the same folder, so the user can open old and new side by side; `.gitignore` keeps it out of commits. Delete it when the user signs off, or when the render is dropped or redone. Then read the diff from the candidate's `meta.repository.revision` to `HEAD` over the page's source paths and edit that spec the way the update step edits a page. Change only the nodes, flows, cards, and source citations the diff touches, keep every position and route that still holds, and set the revision to `HEAD`'s full commit. Fix any line range `finalize` reports out of range at the new revision. Done when `finalize` passes and the spec's own diff shows only what changed in the system.
3. **Invoke the archify skill** with a request built from the page:
   - the topic and the type you chose
   - the page's summary and "How it works" text, as the description of what to draw
   - the reader: an engineer learning this system well enough to explain it
   - that this is a real repository, so archify attaches source evidence
   - the page's Mermaid diagram as the statement of the topic's main claims: the rendered diagram must agree with it on what the pieces are, how they relate, and which boundaries they sit in. Layout is archify's choice, and it may add or leave out detail around those claims.
4. **Apply these layout rules** while writing the request and editing the spec. They are the rules that prevented a `finalize` or visual-check failure on earlier renders; tally any rule you add against a real failure before keeping it.

   **Before coordinates.** Pick the size from the content, then write positions. An architecture diagram shows the whole graph at 1440 × 900 in the default view; fold detail into sublabels and cards rather than adding a row. A region's default pad is 30px, so a region whose member sits near the canvas edge overflows it; set a smaller `pad` on a region that touches an edge, and give each region its own columns since a region is the bounding box of what it wraps. A sequence diagram that passes the readability floor holds only about a dozen messages, so choose the messages before writing coordinates.

   **While editing.** Keep the canvas fixed and fix text by shortening labels or moving the fact to a card; a wider canvas fails the readability floor instead of helping. Let archify route forward flows; give a return flow explicit `via` points outside the rows plus `fromSide`/`toSide`, or it runs along a region border. Put a per-node setting such as the model variable in `tag`, which shows on detail, so the sublabel carries the role. A region label or `tag` can replace an edge where it conveys the same fact. The architecture schema caps `sources` at 3 per component: pick the three that prove the box's claims and leave the rest for a card. In a sequence diagram, put explanation in cards, not in message notes, since a note costs a row the canvas does not have; keep segment labels short. Set `meta.legend.entries` whenever a variant means something other than archify's default.

   **Before finishing.** `finalize` refuses to overwrite `<topic-slug>.browser-check.json`, so delete it by name between runs. Delete every receipt (`.finalize.json`, `.finalize-summary.json`, `.browser-check.json`, `.delivery.json`) before commit so the folder holds the spec and the HTML only. Run `visual-check <html> --summary --require-provenance --out-dir <scratch>` and read the 1440 × 900 PNG yourself. It is the only check of the whole-graph rule and it catches what `finalize` passes: an oversized graph, a label hidden behind the next one, a label past the right edge or touching a region border.
5. **Finish** when archify's `finalize` passes and the 1440 × 900 capture shows the whole graph, readable labels, and nothing hidden. Passing `finalize` does not establish that. If the capture could not be produced or viewed, say so plainly instead of claiming visual quality.

Add the diagram to the table in `docs/architecture/diagrams/README.md` (create it on the first render: one row per diagram with the topic, the type, and the HTML path) and a "Rendered with archify" link under the topic page's Diagram section, if not already there, and check the link resolves. `.archify/` is ignored and is only for scratch runs.

Then commit, push, and open a draft PR as `AGENTS.md` (Pull requests) describes, titled `docs(architecture): render the <topic-slug> diagram` (`re-render` for a re-render), or push to the PR this run already has. Stop and give the user the absolute path of the new HTML, of the `-previous` file if there is one, and the PR link, with the viewport you checked at and anything archify's visual review recommendation flagged. Wait for their visual check.
