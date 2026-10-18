from backend.report import generate_report
import os
import uuid
import secrets
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from backend.worker import analyze_video_file

app = FastAPI(
    title="Lighthouse-AI",
    version="0.3.0",
)

UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("LIGHTHOUSE_API_KEY")

MAX_FILE_SIZE = 100 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024

jobs = {}
jobs_lock = Lock()

executor = ThreadPoolExecutor(max_workers=1)


def verify_api_key(key: Optional[str]):
    if not API_KEY:
        raise HTTPException(503, "API key not configured")

    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(401, "Invalid API key")


def process_job(job_id: str, file_path: str):
    try:
        result = analyze_video_file(file_path, job_id=job_id)

        with jobs_lock:
            jobs[job_id].update({
                "status": "done",
                "verdict": result["verdict"],
                "violations": result["violations"],
                "diagnostic": result["diagnostic"],
            })

    except Exception as error:
        with jobs_lock:
            jobs[job_id].update({
                "status": "failed",
                "error": str(error),
            })

    finally:
        Path(file_path).unlink(missing_ok=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Lighthouse-AI",
        "ai_ready": False,
    }


@app.post("/api/v1/analyze")
async def analyze(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    verify_api_key(x_api_key)

    if Path(file.filename or "").suffix.lower() != ".mov":
        raise HTTPException(400, "Only .mov supported")

    job_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{job_id}.mov"

    size = 0

    try:
        with path.open("wb") as output:
            while True:
                chunk = await file.read(CHUNK_SIZE)

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_FILE_SIZE:
                    raise HTTPException(413, "File too large")

                output.write(chunk)

        if size == 0:
            raise HTTPException(400, "Empty file")

    except Exception:
        path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "verdict": None,
            "violations": [],
        }

    executor.submit(process_job, job_id, str(path))

    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}")
def get_job(
    job_id: str,
    x_api_key: Optional[str] = Header(default=None),
):
    verify_api_key(x_api_key)

    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(404, "Job not found")

        return dict(job)



@app.get("/api/v1/jobs/{job_id}/report")
def get_job_report(
    job_id: str,
    x_api_key: Optional[str] = Header(default=None),
):
    verify_api_key(x_api_key)

    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Job not found",
            )

        status = job.get("status")

        if status != "done":
            return {
                "job_id": job_id,
                "status": status,
                "ready": False,
            }

        diagnostic = dict(job.get("diagnostic") or {})

    return generate_report(job_id, diagnostic)
