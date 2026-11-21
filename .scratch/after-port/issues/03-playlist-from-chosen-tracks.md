# 03: A playlist from tracks the user picked

**Status:** needs-planning
**Planning approach:** grill-with-docs

**Planning gate:** Refine against the current implementation with the user before
implementation. Update scope, acceptance criteria, and dependencies; record
settled domain terms and architectural decisions where needed. Return to
ready-for-agent once the user agrees the plan is settled.

**What is missing:** `build_mood_playlist` builds from the tracks stored under a mood, and
takes no track list. Asked to "build a playlist from the top 5" retrieved tracks, the
assistant can only offer to add them to a playlist the user already has
(`add_tracks_to_playlist`) or to build a mood playlist without them. The retrieved tracks
carry Spotify ids, so a new playlist from a given list of ids needs no lookup.

**To settle:** a new tool or a new argument on the existing one.
