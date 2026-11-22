# 07: The assistant hedges, and stops claiming band activity

**Spec:** `.scratch/ml-rebuild/spec.md` (The assistant)

**What to build:** When the assistant analyzes a session it says, in one short clause, how reliable the reading is, worded by the loaded checkpoint's single number. Its explanation of a quadrant says what the quadrant means and no longer claims what the listener's frequency bands did, because nothing measured them for that session.

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] The sessions capability's instructions carry the checkpoint's single number (the mean over the participants it holds) and one wording tier: below 0.60 "a rough read", 0.60 to 0.75 "a fair read", 0.75 and above no hedge
- [ ] The instructions ask for one short clause, once per analysis, never a paragraph
- [ ] With no checkpoint loaded the most cautious tier is used
- [ ] The hedge text is fixed for the life of the process and sits in the cached prefix; brain context stays the only per-request part (ADR 0002)
- [ ] The four canned quadrant explanations say what each quadrant means, with no claim about band activity
- [ ] The system prompt's quadrant list loses its band claims, and its band line lists theta, alpha, beta and gamma with the recipe's edges
- [ ] Unit tests cover the tier at each cut-off and with no checkpoint
- [ ] The eval tier is not run for this ticket; note in the PR that the narration and explanation cases may shift
