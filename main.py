"""
Nooi Floor Plan Analysis Service
No AI APIs — pure OpenCV + Tesseract OCR.
"""
import logging, os, traceback
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import analyse_floor_plan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Nooi Floor Plan Service", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AnalyseRequest(BaseModel):
    image_url:    str
    project_id:   str
    gemini_api_key: Optional[str] = None  # kept for API compatibility, not used


class AnalyseResponse(BaseModel):
    success:    bool
    rooms:      list
    walls:      list
    openings:   list
    image_size: dict
    error:      Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "nooi-floorplan-service", "version": "2.0.0"}


@app.post("/analyse", response_model=AnalyseResponse)
async def analyse(req: AnalyseRequest):
    log.info(f"Analyse: project={req.project_id}")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(req.image_url)
            if resp.status_code != 200:
                raise HTTPException(502, f"Cannot fetch image: {resp.status_code}")
            image_bytes = resp.content

        result = await analyse_floor_plan(
            image_bytes=image_bytes,
            image_url=req.image_url,
            project_id=req.project_id,
            gemini_api_key=req.gemini_api_key or os.getenv("GEMINI_API_KEY", ""),
        )
        log.info(f"Done: {len(result['rooms'])} rooms, {len(result['walls'])} walls, {len(result['openings'])} openings")
        return AnalyseResponse(success=True, **result)
    except HTTPException:
        raise
    except Exception as exc:
        log.error(traceback.format_exc())
        return AnalyseResponse(
            success=False, rooms=[], walls=[], openings=[],
            image_size={"width": 0, "height": 0}, error=str(exc),
        )