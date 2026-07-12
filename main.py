import logging
import os
import traceback
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import analyse_floor_plan

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Nooi Floor Plan Service", version="3.0.0")

# CORS: this service is only ever called server-to-server by the Express
# backend, so no browser origins are needed at all.
_allowed = [o for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o]
app.add_middleware(CORSMiddleware, allow_origins=_allowed,
                   allow_methods=["POST", "GET"], allow_headers=["*"])

# Shared secret: set SERVICE_KEY on Railway for BOTH this service and the
# Express backend, and send it as the X-Service-Key header from Express.
SERVICE_KEY = os.getenv("SERVICE_KEY", "")

# Only fetch floor plan images from our own storage (SSRF guard). Set to your
# Supabase project host, e.g. "abcdefgh.supabase.co". Comma-separate to allow
# several. Empty = allow any https host (NOT recommended in production).
ALLOWED_IMAGE_HOSTS = [h.strip().lower()
                       for h in os.getenv("ALLOWED_IMAGE_HOSTS", "").split(",")
                       if h.strip()]


def _check_image_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme != "https":
        raise HTTPException(400, "image_url must be https")
    host = (p.hostname or "").lower()
    if ALLOWED_IMAGE_HOSTS and not any(
            host == a or host.endswith("." + a) for a in ALLOWED_IMAGE_HOSTS):
        raise HTTPException(400, "image_url host not allowed")


class AnalyseRequest(BaseModel):
    image_url: str
    project_id: str
    gemini_api_key: Optional[str] = None   # legacy field, ignored


class AnalyseResponse(BaseModel):
    success: bool
    rooms: list          # each room now includes `polygon`: [[x%, y%], ...]
    walls: list          # centerline segments, % coords + thickness
    openings: list       # {type, wall_id, wall, x‰, y‰, width‰}
    image_size: dict
    scale_m_per_px: Optional[float] = None
    error: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "nooi-floorplan-service",
            "version": "3.0.0"}


@app.post("/analyse", response_model=AnalyseResponse)
async def analyse(req: AnalyseRequest,
                  x_service_key: str = Header(default="")):
    if SERVICE_KEY and x_service_key != SERVICE_KEY:
        raise HTTPException(401, "invalid service key")
    _check_image_url(req.image_url)

    log.info(f"Analyse: project={req.project_id}")
    try:
        async with httpx.AsyncClient(timeout=30,
                                     follow_redirects=False) as client:
            resp = await client.get(req.image_url)
            if resp.status_code != 200:
                raise HTTPException(502, f"Cannot fetch image: {resp.status_code}")
            if len(resp.content) > 15 * 1024 * 1024:
                raise HTTPException(413, "image too large")
            image_bytes = resp.content

        result = await analyse_floor_plan(
            image_bytes=image_bytes,
            image_url=req.image_url,
            project_id=req.project_id,
        )
        log.info(f"Done: {len(result['rooms'])} rooms, "
                 f"{len(result['walls'])} walls, "
                 f"{len(result['openings'])} openings")
        return AnalyseResponse(success=True, **result)
    except HTTPException:
        raise
    except Exception as exc:
        log.error(traceback.format_exc())
        return AnalyseResponse(
            success=False, rooms=[], walls=[], openings=[],
            image_size={"width": 0, "height": 0}, error=str(exc),
        )