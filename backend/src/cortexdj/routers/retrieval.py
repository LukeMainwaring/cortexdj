from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from cortexdj.dependencies.db import AsyncPostgresSessionDep
from cortexdj.schemas.retrieval import SimilarTrackSchema, SimilarTracksResponse
from cortexdj.services import retrieval as retrieval_service
from cortexdj.services.retrieval import DeapFileMissingError

retrieval_router = APIRouter(prefix="/sessions", tags=["retrieval"])


@retrieval_router.get("/{session_id}/similar-tracks")
async def get_similar_tracks(
    db: AsyncPostgresSessionDep,
    session_id: str,
    k: int = Query(default=10, ge=1, le=100),
) -> SimilarTracksResponse:
    """Return the top-k tracks whose CLAP audio embedding is closest to the
    session's EEG embedding in the joint contrastive space."""
    try:
        hits = await retrieval_service.retrieve_similar_tracks(db, session_id, k=k)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DeapFileMissingError as exc:
        # Server misconfig (session references missing DEAP data) — not 503.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        # Reserved for missing contrastive encoder checkpoint only.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SimilarTracksResponse(
        session_id=session_id,
        tracks=[
            SimilarTrackSchema(
                spotify_id=h.spotify_id,
                title=h.title,
                artist=h.artist,
                itunes_preview_url=h.itunes_preview_url,
                similarity=h.similarity,
            )
            for h in hits
        ],
        k=k,
    )
