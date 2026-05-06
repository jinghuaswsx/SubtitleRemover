"""FastAPI service for SubtitleRemover.

Endpoints:
    GET  /health               — liveness probe
    POST /remove-subtitle      — upload video, returns task_id
    GET  /status/{task_id}     — poll task progress
    GET  /download/{task_id}   — download the cleaned video

Tasks run in a single background worker thread (one job at a time, since the
pipeline contends for GPU/IO). This is good enough for Phase B; Phase C may
upgrade to a proper queue if concurrent jobs become necessary.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .pipeline import process_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("api")


def _apply_gpu_fraction() -> None:
    """Cap this process at SR_GPU_FRACTION of total VRAM. Must run before any
    CUDA op so the limit applies to all subsequent allocations."""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("CUDA unavailable; SR_GPU_FRACTION ignored")
            return
        frac = max(0.05, min(1.0, config.GPU_MEMORY_FRACTION))
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("GPU fraction capped at %.2f → %.2f GiB of %.2f GiB",
                    frac, total * frac, total)
    except Exception as e:  # never block startup over an advisory cap
        logger.warning("could not apply GPU fraction: %s", e)


_apply_gpu_fraction()


@dataclass
class Task:
    task_id: str
    input_path: str
    output_path: str
    roi_spec: Optional[str]
    detection: str = "auto"
    ocr_engine: str = "easyocr"
    inpaint: str = "opencv"
    vsr: str = "off"
    state: str = "queued"  # queued / running / done / failed
    progress: float = 0.0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


TASKS: Dict[str, Task] = {}
QUEUE: "Queue[str]" = Queue()


def _worker() -> None:
    while True:
        task_id = QUEUE.get()
        task = TASKS.get(task_id)
        if task is None:
            continue
        task.state = "running"
        logger.info("task %s started (input=%s)", task_id, task.input_path)
        try:
            def cb(p: float) -> None:
                task.progress = p
            process_video(
                task.input_path,
                task.output_path,
                detection=task.detection,
                ocr_engine=task.ocr_engine,
                roi_spec=task.roi_spec,
                inpaint=task.inpaint,
                vsr=task.vsr,
                progress_cb=cb,
            )
            task.state = "done"
            task.progress = 1.0
            logger.info("task %s done", task_id)
        except Exception as e:  # surface to caller via /status
            task.state = "failed"
            task.error = str(e)
            logger.exception("task %s failed", task_id)
        finally:
            task.finished_at = time.time()


threading.Thread(target=_worker, daemon=True, name="sr-worker").start()


app = FastAPI(title="SubtitleRemover", version="0.2.0-phaseC")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "phase": "C",
        "queued": QUEUE.qsize(),
        "tasks_in_memory": len(TASKS),
    }


_ALLOWED_DETECTION = {"auto", "roi", "ocr"}
_ALLOWED_OCR = {"easyocr", "paddle"}
_ALLOWED_INPAINT = {"opencv", "lama"}
_ALLOWED_VSR = {"off", "none", "real-esrgan", "realesrgan",
                "basicvsr++", "basicvsrpp", "basicvsr"}


@app.get("/info")
def info() -> dict:
    return {
        "version": app.version,
        "detection": sorted(_ALLOWED_DETECTION),
        "ocr_engine": sorted(_ALLOWED_OCR),
        "inpaint": sorted(_ALLOWED_INPAINT),
        "vsr": sorted(_ALLOWED_VSR),
        "max_resolution": [config.MAX_WIDTH, config.MAX_HEIGHT],
        "max_duration_min": config.MAX_DURATION_MIN,
        "supported_inputs": config.SUPPORTED_INPUTS,
        "recommended_best_quality": {
            "detection": "ocr",
            "ocr_engine": "easyocr",
            "inpaint": "lama",
            "vsr": "real-esrgan",
        },
    }


@app.post("/remove-subtitle")
async def remove_subtitle(
    file: UploadFile = File(...),
    roi: Optional[str] = Form(default=None),
    detection: str = Form(default="auto"),
    ocr_engine: str = Form(default="easyocr"),
    inpaint: str = Form(default="opencv"),
    vsr: str = Form(default="off"),
) -> JSONResponse:
    if detection not in _ALLOWED_DETECTION:
        raise HTTPException(400, f"detection must be one of {_ALLOWED_DETECTION}")
    if ocr_engine not in _ALLOWED_OCR:
        raise HTTPException(400, f"ocr_engine must be one of {_ALLOWED_OCR}")
    if inpaint not in _ALLOWED_INPAINT:
        raise HTTPException(400, f"inpaint must be one of {_ALLOWED_INPAINT}")
    if vsr not in _ALLOWED_VSR:
        raise HTTPException(400, f"vsr must be one of {_ALLOWED_VSR}")

    ext = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
    if ext not in {fmt.lower() for fmt in config.SUPPORTED_INPUTS}:
        raise HTTPException(415, f"unsupported format: .{ext}")

    task_id = uuid.uuid4().hex
    input_path = os.path.join(config.UPLOAD_DIR, f"{task_id}.{ext}")
    output_path = os.path.join(config.OUTPUT_DIR, f"{task_id}.mp4")

    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    task = Task(task_id=task_id, input_path=input_path,
                output_path=output_path, roi_spec=roi,
                detection=detection, ocr_engine=ocr_engine,
                inpaint=inpaint, vsr=vsr)
    TASKS[task_id] = task
    QUEUE.put(task_id)
    logger.info("task %s queued (file=%s, detection=%s/%s, inpaint=%s, vsr=%s, roi=%s)",
                task_id, file.filename, detection, ocr_engine, inpaint, vsr, roi)
    return JSONResponse({"task_id": task_id, "state": "queued"})


@app.get("/status/{task_id}")
def status(task_id: str) -> dict:
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, "task not found")
    return {
        "task_id": task.task_id,
        "state": task.state,
        "progress": round(task.progress, 3),
        "error": task.error,
        "created_at": task.created_at,
        "finished_at": task.finished_at,
    }


@app.get("/download/{task_id}")
def download(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, "task not found")
    if task.state != "done":
        raise HTTPException(409, f"task not done (state={task.state})")
    if not os.path.exists(task.output_path):
        raise HTTPException(410, "output file missing")
    return FileResponse(task.output_path, media_type="video/mp4",
                        filename=f"{task_id}.mp4")


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=config.SERVICE_PORT, log_level="info")


if __name__ == "__main__":
    main()
