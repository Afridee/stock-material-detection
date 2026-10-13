"""HTTP API tests. No model is loaded; ``run_detect`` is stubbed."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from stock_material_detection.api import app
from stock_material_detection.pipeline import CountLine, DetectResult


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STOCK_WARMUP", "0")
    monkeypatch.delenv("SERVICE_API_KEY", raising=False)

    def _fake_detect(image, **kwargs):
        return DetectResult(
            image=str(image),
            counts=[CountLine(sku="unknown", category="cigarette_pack", count=1, flagged=True)],
            detections=[],
        )

    monkeypatch.setattr("stock_material_detection.api.run_detect", _fake_detect)
    with TestClient(app) as test_client:
        yield test_client


def _photo() -> dict:
    return {"image": ("photo.jpg", io.BytesIO(b"not-a-real-jpeg"), "image/jpeg")}


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_detect_returns_cli_payload(client: TestClient):
    response = client.post("/detect", files=_photo())
    assert response.status_code == 200
    body = response.json()
    assert body["image"] == "photo.jpg"
    assert body["counts"] == [
        {"sku": "unknown", "category": "cigarette_pack", "count": 1, "flagged": True}
    ]
    assert body["detections"] == []
    assert "failReasons" not in body


def test_detect_rejects_non_image(client: TestClient):
    files = {"image": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    assert client.post("/detect", files=files).status_code == 422


def test_detect_rejects_empty_image(client: TestClient):
    files = {"image": ("photo.jpg", io.BytesIO(b""), "image/jpeg")}
    assert client.post("/detect", files=files).status_code == 422


def test_detect_requires_api_key_when_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    assert client.post("/detect", files=_photo()).status_code == 401

    allowed = client.post("/detect", files=_photo(), headers={"X-API-Key": "secret"})
    assert allowed.status_code == 200
