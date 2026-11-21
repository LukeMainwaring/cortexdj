# 01: Decode the cached `.m4a` previews

**Status:** needs-planning
**Planning approach:** grill-with-docs

**Planning gate:** Refine against the current implementation with the user before
implementation. Update scope, acceptance criteria, and dependencies; record
settled domain terms and architectural decisions where needed. Return to
ready-for-agent once the user agrees the plan is settled.

**What is wrong:** `seed-track-index` cannot build the track index. It decodes each
track's iTunes preview with `load_audio_waveform` in `cortexdj/ml/contrastive.py`,
which calls `librosa.load`, and that raises `LibsndfileError: Format not recognised`
on every cached preview. The previews are AAC audio in `.m4a` files. libsndfile does
not read AAC, and librosa 1.0 removed the `audioread` fallback that used to pass
such files to ffmpeg, so having ffmpeg installed no longer helps. Seen with librosa
1.0.1 and soundfile 0.14.0, on the host and in the backend container.

**What it blocks:** adding tracks to the index, and building the index from nothing.
Retrieval itself is not affected: it reads stored vectors and decodes no audio. The
index in use was built before this broke and is kept as
`backend/data/track-index.dump` (447 tracks, data only, one table). A fresh clone has
no such file, so until this is fixed a new setup has no track index and
`retrieve_tracks_from_brain_state` has nothing to search.

**Take this up with** the next audio or ML work, or the next upgrade of librosa,
soundfile, torch, or transformers, since any of those may change the answer.

## Two ways to decode, both tried on one cached preview

Each gave the same result on 2026-11-20: 1,436,596 mono float32 samples at 48 kHz
from a 29.98 s stereo 44.1 kHz AAC file.

1. **PyAV** (the `av` package, 19.0.2 tried): decode and resample to mono 48 kHz in
   Python. Its wheels carry their own ffmpeg libraries, so it behaves the same on
   the host, in the container, and in CI. Cost: one new runtime dependency.
2. **The `ffmpeg` program**: run `ffmpeg -i <file> -f f32le -ac 1 -ar 48000 -` and
   read the bytes into numpy. No new dependency, but the program must be installed
   wherever the script runs. The backend image (`python:3.14-slim-bookworm`) does
   not have it.

## Not yet checked

- Whether either raises a warning under this repo's warnings-as-errors test setting.
- Whether either matches, sample for sample, what the old librosa path produced.
  Vectors already in the index came from that path. If the new decoder differs,
  new tracks are embedded slightly differently from old ones unless the whole index
  is rebuilt.
- Whether the rest of `seed-track-index` still works once decoding does: the Spotify
  candidate fetch, the iTunes preview lookup, and the CLAP encoder have not run
  since before the librosa upgrade either.

## Acceptance criteria (provisional until planning)

- [ ] `load_audio_waveform` decodes a cached `.m4a` preview to mono 48 kHz float32, with a test on a small real AAC file
- [ ] `seed-track-index` runs to the end on the host against the audio cache
- [ ] The setup docs say how a fresh clone gets a track index, with the measured run time
- [ ] The index is either rebuilt whole with the new decoder, or the planning session recorded why mixing decoders is acceptable
