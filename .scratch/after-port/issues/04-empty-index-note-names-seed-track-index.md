# 04: The empty-index note tells the reader to run `seed-track-index`

**Status:** ready-for-agent
**Blocked by:** audio-decoding 01, unless the note is reworded to the dump

**What is wrong:** when the track index is empty, `retrieve_tracks_from_brain_state`
(`features/retrieval/tools.py`) returns a `note` that says to run
`uv run --directory backend seed-track-index`. That command cannot finish today and deletes
its miss log under `backend/data/` as soon as it has a Spotify client. `DEVELOPMENT.md`
says the index is restored from `backend/data/track-index.dump`.

**What to build:** the note points at the restore in `DEVELOPMENT.md` for as long as
`seed-track-index` cannot run. The tool's docstring and any test that reads the note follow.

## Acceptance criteria

- [ ] The note names no command that deletes data
- [ ] The test for the empty index asserts the new wording
