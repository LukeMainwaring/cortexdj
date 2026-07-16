# iTunes Search previews instead of Spotify `preview_url`

**Status**: Accepted — 2026-07-16

Spotify deprecated `preview_url` for standard-mode apps on 2024-11-27; against
this project's Spotify app it now returns 0/10 hits, so it cannot supply the 30s
audio the CLAP encoder needs. `services/audio_catalog.py` resolves previews from
the iTunes Search API instead, and Spotify remains the source of truth for track
identity — an iTunes hit is only accepted for a track Spotify already named.

Because iTunes matches on a text query, a hit can be the wrong edit of the right
song. Spotify's `duration_ms` anchors the match: candidates outside a 3s
duration delta are rejected outright, and survivors are ranked by artist
similarity, then title similarity, then smallest delta. Do not drop the duration
filter in favor of text similarity alone — remasters and live versions score
identically on title.
