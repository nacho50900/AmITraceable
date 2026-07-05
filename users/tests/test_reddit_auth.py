"""
Tests del módulo de autenticación con Reddit.

Las llamadas HTTP reales a Reddit (intercambio de code -> token) se mockean
con `respx` sobre `httpx`, igual que se haría con cualquier integración
externa: no dependemos de credenciales reales ni de red para verificar que
el flujo está bien construido.
"""
import httpx
import respx
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_reddit_status_defaults_to_unauthenticated():
    resp = client.get("/auth/reddit/status")
    assert resp.status_code == 200
    assert resp.json() == {"authenticated": False}


def test_reddit_login_redirects_with_expected_scopes():
    resp = client.get("/auth/reddit/login", follow_redirects=False)
    assert resp.status_code == 307
    location = resp.headers["location"]
    assert location.startswith("https://www.reddit.com/api/v1/authorize")
    assert "scope=identity+history+read" in location


def test_reddit_callback_rejects_invalid_state():
    resp = client.get(
        "/auth/reddit/callback",
        params={"code": "fake-code", "state": "not-the-saved-state"},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_reddit_callback_redirects_on_user_denied_consent():
    resp = client.get(
        "/auth/reddit/callback",
        params={"error": "access_denied"},
        follow_redirects=False,
    )
    assert resp.status_code == 307
    assert "auth_error=access_denied" in resp.headers["location"]


@respx.mock
def test_reddit_callback_exchanges_token_and_sets_session():
    # 1. Forzamos un estado válido en la sesión simulando el paso de /login
    session_client = TestClient(app)
    login_resp = session_client.get("/auth/reddit/login", follow_redirects=False)
    state = login_resp.headers["location"].split("state=")[1].split("&")[0]

    # 2. Mockeamos el intercambio code -> access_token
    respx.post("https://www.reddit.com/api/v1/access_token").mock(
        return_value=httpx.Response(200, json={"access_token": "fake-access-token", "token_type": "bearer"}),
    )

    callback_resp = session_client.get(
        "/auth/reddit/callback",
        params={"code": "fake-code", "state": state},
        follow_redirects=False,
    )

    assert callback_resp.status_code == 307
    assert "/dashboard" in callback_resp.headers["location"]

    status_resp = session_client.get("/auth/reddit/status")
    assert status_resp.json() == {"authenticated": True}


def test_reddit_logout_clears_only_reddit_session_keys():
    session_client = TestClient(app)
    with session_client as c:
        c.get("/auth/reddit/login")  # crea la cookie de sesión inicial
        resp = c.post("/auth/reddit/logout")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

        status_resp = c.get("/auth/reddit/status")
        assert status_resp.json() == {"authenticated": False}


def test_analyze_reddit_requires_authentication():
    resp = client.post("/api/analyze/reddit")
    assert resp.status_code == 401
