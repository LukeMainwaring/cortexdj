import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from cortexdj.core.db import AsyncPostgresSessionDep
from cortexdj.core.paths import AUDIO_CACHE_DIR
from cortexdj.features.retrieval import service as retrieval_service
from cortexdj.features.retrieval.encoder import ContrastiveEncoder, get_contrastive_encoder
from cortexdj.features.retrieval.schema import SimilarTrackSchema, SimilarTracksResponse

retrieval_router = APIRouter(prefix="/sessions", tags=["retrieval"])
audio_router = APIRouter(prefix="/audio", tags=["audio"])

ContrastiveEncoderDep = Annotated[ContrastiveEncoder | None, Depends(get_contrastive_encoder)]

# SHA-1 hex (matches audio_catalog.cache_key). Anchored to prevent
# any path-traversal segment from sneaking through the {cache_key} param.
_CACHE_KEY_RE = re.compile(r"^[a-f0-9]{40}$")


@retrieval_router.get("/{session_id}/similar-tracks")
async def get_similar_tracks(
    db: AsyncPostgresSessionDep,
    encoder: ContrastiveEncoderDep,
    session_id: str,
    k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> SimilarTracksResponse:
    """Return the top-k tracks whose CLAP audio embedding is closest to the
    session's EEG embedding in the joint contrastive space."""
    try:
        hits = await retrieval_service.retrieve_similar_tracks(db, session_id, encoder, k=k)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except retrieval_service.DeapFileMissingError as exc:
        # Server misconfig (session references missing DEAP data) — not 503.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except retrieval_service.ContrastiveEncoderMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SimilarTracksResponse(
        session_id=session_id,
        tracks=[
            SimilarTrackSchema(
                spotify_id=h.spotify_id,
                title=h.title,
                artist=h.artist,
                itunes_preview_url=h.itunes_preview_url,
                audio_cache_key=h.audio_cache_key,
                similarity=h.similarity,
            )
            for h in hits
        ],
        k=k,
    )


@audio_router.get("/preview/{cache_key}")
async def get_audio_preview(cache_key: str) -> FileResponse:
    """Serve a cached iTunes m4a preview by its content-addressed cache key.

    Same-origin delivery exists so the frontend waveform component can
    `fetch` + `decodeAudioData` the bytes — Apple's preview CDN does not
    reliably set CORS headers for cross-origin Web Audio decoding.
    """
    if not _CACHE_KEY_RE.match(cache_key):
        raise HTTPException(status_code=400, detail="invalid cache key")

    path = AUDIO_CACHE_DIR / f"{cache_key}.m4a"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="preview not cached")

    return FileResponse(path, media_type="audio/mp4")
