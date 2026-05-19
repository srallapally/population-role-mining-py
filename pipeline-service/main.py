# pipeline-service/main.py
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

import config
from pipeline.orchestrator import run_pipeline, PipelineCancelled
from es import client as es_client

import os
from fastapi import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("pipeline-service")

_PIPELINE_KEY = os.environ.get("PIPELINE_API_KEY", "")


class RunRequest(BaseModel):
    sessionId: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        es_client.get(config.INDEX_SESSIONS, "__healthcheck_nonexistent__")
        logger.info("ES reachable at %s", config.ES_HOST)
    except Exception as exc:
        logger.warning("ES unreachable at %s: %s", config.ES_HOST, exc)
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if _PIPELINE_KEY and request.url.path != "/health":
        if request.headers.get("X-Pipeline-Key") != _PIPELINE_KEY:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing X-Pipeline-Key"},
            )
    return await call_next(request)


@app.post("/run")
def run(req: RunRequest) -> dict:
    try:
        run_pipeline(req.sessionId)
        return {"status": "complete", "sessionId": req.sessionId}
    except PipelineCancelled:
        return {"status": "cancelled", "sessionId": req.sessionId}
    except Exception as exc:
        return {"status": "failed", "sessionId": req.sessionId, "error": str(exc)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PIPELINE_PORT)
