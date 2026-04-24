# README media — capture checklist

Media referenced from [`README.md`](../../README.md). Recorded manually; drop the files here with the exact names below.

Suggested capture settings: 1920×1080 screen, app at default zoom, dark-mode UI, session seed already loaded. Target <5 MB per GIF (use [`gifsicle`](https://www.lcdf.org/gifsicle/) `-O3` or [`ffmpeg -vf palettegen`](https://ffmpeg.org/) if big). Samplespace's `docs/assets/*.gif` are a good size reference.

| File | Type | Width in README | What it should show |
|------|------|-----------------|----------------------|
| `end-to-end-session-to-playlist.gif` | Hero GIF | 700 px | Full headline demo: user asks "Analyze my most recent EEG session and build a playlist." Agent runs `list_sessions` → `analyze_session` (trajectory renders) → `find_relaxing_tracks` → proposes playlist → user confirms → `build_mood_playlist` completes. Keep it under ~15 s. |
| `session-trajectory.png` | Screenshot | 90 % | `analyze_session` panel with the animated SVG trajectory visible (mid-animation is fine), timeline tab accessible, and the stacked band-power chart below. Dark mode. |
| `retrieved-tracks-panel.png` | Screenshot | 90 % | `retrieve_tracks_from_brain_state` output: ranked track list with visible cosine-similarity bars (include at least one emerald ≥0.5, one amber ≥0.25, one grey), play button and Spotify external-link button on each row. |
| `brain-grounded-playlist.gif` | GIF | 700 px | Prompt "Build me a playlist of tracks that triggered relaxed brain states." Capture the `find_relaxing_tracks` tool-call panel expanding, the agent's proposal, the user confirmation, and the final Spotify-playlist success message. |
| `brain-context-persistence.gif` | GIF | 700 px | Two-step flow: (1) user sets brain context for Session 11, pill appears in chat header; (2) page refresh — pill survives; (3) user asks "build me a playlist" and the agent's response references the persisted context without being reminded. |

Once captured, delete this file or trim it down — it's scaffolding for the initial capture, not long-term docs.
