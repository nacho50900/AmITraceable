"""
Tests de los endpoints HTTP principales.

Equivalente en pytest+httpx al test original `users-service.test.js`
(supertest sobre la app de Express), pero adaptado a FastAPI/TestClient.
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


def test_auth_status_defaults_to_unauthenticated():
    resp = client.get("/auth/reddit/status")
    assert resp.status_code == 200
    assert resp.json() == {"authenticated": False}


def test_login_redirects_to_reddit_with_expected_scopes():
    resp = client.get("/auth/reddit/login", follow_redirects=False)
    assert resp.status_code == 307
    location = resp.headers["location"]
    assert location.startswith("https://www.reddit.com/api/v1/authorize")
    assert "scope=identity+history+read" in location


def test_analyze_requires_authentication():
    resp = client.post("/api/analyze")
    assert resp.status_code == 401


def test_metrics_endpoint_is_exposed():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"http_requests" in resp.content or b"# HELP" in resp.content
