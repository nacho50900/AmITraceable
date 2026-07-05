"""
Tests de los endpoints HTTP genéricos de la aplicación (no específicos de
ninguna plataforma). Los tests de autenticación/análisis de cada plataforma
viven en su propio archivo simétrico: `test_reddit_auth.py` /
`test_reddit_client.py` y `test_instagram_auth.py` / `test_instagram_client.py`.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_returns_service_info():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "identity-exposure-tfg"


def test_metrics_endpoint_is_exposed():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"http_requests" in resp.content or b"# HELP" in resp.content


def test_analyze_rejects_unsupported_platform():
    resp = client.post("/api/analyze/tiktok")
    assert resp.status_code == 404
    assert "tiktok" in resp.json()["detail"]
