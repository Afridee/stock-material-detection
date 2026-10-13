"""HTTP API for stock material detection.

Same pipeline as ``stock_material_detection detect``. Run one process:

    uv run python -m uvicorn stock_material_detection.api:app --host 0.0.0.0 --port 8000
    POST /detect
      multipart field ``image`` — JPEG or PNG photo, max 25MB

Authenticate with header ``X-API-Key: <SERVICE_API_KEY>``. The check runs
only when that env var is set, so local calls work without it.

The detector and gallery load once per process. Inference is serialized:
one photo is in the model at a time. Do not add uvicorn workers.
"""

from __future__ import annotations

# Before pipeline, which imports cv2 and torch. OpenMP latches its thread
# count on that first native import.
from stock_material_detection.cpu_threads import configure_cpu_threads

configure_cpu_threads()

import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi

from stock_material_detection.pipeline import get_detector, get_gallery, get_ocr, run_detect

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("stock_material_detection.api")

MAX_IMAGE_BYTES = 25 * 1024 * 1024
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SUFFIX_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
}

# YOLOE, SigLIP 2, and PaddleOCR are not safe to call concurrently.
_infer_lock = threading.Lock()


def _warmup_enabled() -> bool:
    return os.getenv("STOCK_WARMUP", "1") != "0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _warmup_enabled():
        logger.info("loading detector and gallery")
        get_detector()
        gallery = get_gallery()
        if not gallery.empty:
            _ = gallery.embedder
        get_ocr()
        logger.info("models ready")
    else:
        logger.info("STOCK_WARMUP=0 — models load on the first request")
    if not os.getenv("SERVICE_API_KEY", "").strip():
        logger.warning("SERVICE_API_KEY is unset — POST /detect is unauthenticated")
    yield


app = FastAPI(title="Stock Material Detection Service", lifespan=lifespan)


def require_auth(x_api_key: str | None = Header(default=None)) -> None:
    """Accept ``X-API-Key`` when ``SERVICE_API_KEY`` is set. Unset means open."""
    expected = os.getenv("SERVICE_API_KEY", "").strip()
    if not expected or x_api_key == expected:
        return
    logger.warning("rejected request: invalid X-API-Key")
    raise HTTPException(status_code=401, detail="Missing or invalid credentials")


def custom_openapi():
    """Declare the API key so Swagger UI shows an Authorize dialog."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema.setdefault("components", {})["securitySchemes"] = {
        "ApiKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "SERVICE_API_KEY — for the Nest backend and other trusted callers",
        },
    }
    schema["security"] = [{"ApiKey": []}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


def _image_suffix(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix in _ALLOWED_SUFFIXES:
        return ".jpg" if suffix == ".jpeg" else suffix
    content_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    mapped = _SUFFIX_BY_CONTENT_TYPE.get(content_type)
    if mapped:
        return mapped
    raise HTTPException(status_code=422, detail="image must be a JPEG or PNG")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect", dependencies=[Depends(require_auth)])
def detect_image(image: UploadFile = File(..., description="JPEG or PNG photo")) -> dict:
    """Count items in one photo. Response matches the ``detect`` CLI JSON."""
    suffix = _image_suffix(image)
    raw = image.file.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=422, detail="image is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds 25MB")

    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        logger.info("detecting image=%s bytes=%s", image.filename, len(raw))
        with _infer_lock:
            result = run_detect(Path(tmp_name))
    except HTTPException:
        raise
    except Exception:
        logger.exception("detection failed for image=%s", image.filename)
        raise HTTPException(status_code=500, detail="internal error detecting stock")
    finally:
        Path(tmp_name).unlink(missing_ok=True)

    result.image = Path(image.filename).name if image.filename else None
    logger.info(
        "detected image=%s lines=%d detections=%d failReasons=%s",
        result.image,
        len(result.counts),
        len(result.detections),
        result.failReasons,
    )
    return result.public_dict()
