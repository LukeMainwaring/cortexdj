from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from cortexdj.core.paths import AUDIO_CACHE_DIR

audio_router = APIRouter(prefix="/audio", tags=["audio"])

# SHA-1 hex (matches services.audio_catalog.cache_key). Anchored to prevent
# any path-traversal segment from sneaking through the {cache_key} param.
_CACHE_KEY_RE = re.compile(r"^[a-f0-9]{40}$")


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
