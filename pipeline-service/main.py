# pipeline-service/main.py
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

import config
from pipeline.orchestrator import run_pipeline, PipelineCancelled, _now
from es import client as es_client

import os
import threading
from fastapi import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("pipeline-service")

_PIPELINE_KEY = os.environ.get("PIPELINE_API_KEY", "")

_MAX_CONCURRENT_RUNS = int(os.environ.get("MAX_CONCURRENT_RUNS", 5))
_running = threading.Semaphore(_MAX_CONCURRENT_RUNS)
_RUN_TIMEOUT_SECONDS = int(os.environ.get("RUN_TIMEOUT_SECONDS", 300))


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
    acquired = _running.acquire(blocking=False)
    if not acquired:
        return {
            "status": "failed",
            "sessionId": req.sessionId,
            "error": f"Pipeline service at capacity ({_MAX_CONCURRENT_RUNS} concurrent runs)",
        }
    try:
        result = {"status": None, "error": None}

        def _target():
            try:
                run_pipeline(req.sessionId)
                result["status"] = "complete"
            except PipelineCancelled:
                result["status"] = "cancelled"
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = str(exc)

        t = threading.Thread(target=_target)
        t.start()
        t.join(timeout=_RUN_TIMEOUT_SECONDS)

        if t.is_alive():
            # Thread is still running — we cannot kill it, but we can
            # report the timeout. The thread will eventually finish or
            # be killed when the process exits.
            timeout_msg = f"Pipeline exceeded {_RUN_TIMEOUT_SECONDS}s timeout"
            try:
                es_client.update_doc(config.INDEX_SESSIONS, req.sessionId, {
                    "status": "failed",
                    "errorDetail": timeout_msg,
                    "updatedAt": _now(),
                    "lastUpdatedBy": "pipeline",
                })
            except Exception:
                pass  # best effort
            return {
                "status": "failed",
                "sessionId": req.sessionId,
                "error": timeout_msg,
            }

        response = {"status": result["status"], "sessionId": req.sessionId}
        if result["error"] is not None:
            response["error"] = result["error"]
        return response
    finally:
        _running.release()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PIPELINE_PORT)
