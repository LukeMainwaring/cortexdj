# HNSW over IVFFlat for the track-embedding index

**Status**: Accepted — 2026-07-16

The `track_audio_embeddings` cosine-similarity index (pgvector) is created by
migration `77c744e4b096` on a table that is empty at migration time and filled
later by incremental seed passes. IVFFlat needs a training step over existing
rows to build its lists, so an IVFFlat index created here would be trained on
nothing and degrade to a sequential scan until a manual `REINDEX` after every
seed pass. We use HNSW instead: it builds on an empty table and updates its
graph on insert, and gives better recall at this project's ~2k–10k row scale.

`m = 16, ef_construction = 64` are pgvector's defaults and are left untuned;
revisit only if the table grows past ~100k rows, where IVFFlat's smaller build
cost and memory footprint start to matter.
