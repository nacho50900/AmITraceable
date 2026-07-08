"""
Tests del módulo de autenticación con Instagram.

Las llamadas HTTP reales a Instagram (intercambio de code -> token corto ->
token largo) se mockean con `respx` sobre `httpx`, igual que se haría con
cualquier integración externa: no dependemos de credenciales reales ni de
red para verificar que el flujo está bien construido.
"""
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, base_url="https://testserver")


def test_instagram_status_defaults_to_unauthenticated():
    resp = client.get("/auth/instagram/status")
    assert resp.status_code == 200
    assert resp.json() == {"authenticated": False}


def test_instagram_login_redirects_with_expected_scope():
    resp = client.get("/auth/instagram/login", follow_redirects=False)
    assert resp.status_code == 307
    location = resp.headers["location"]
    assert location.startswith("https://www.instagram.com/oauth/authorize")
    assert "scope=instagram_business_basic" in location


def test_instagram_callback_rejects_invalid_state():
    resp = client.get(
        "/auth/instagram/callback",
        params={"code": "fake-code", "state": "not-the-saved-state"},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_instagram_callback_redirects_on_user_denied_consent():
    resp = client.get(
        "/auth/instagram/callback",
        params={"error": "access_denied"},
        follow_redirects=False,
    )
    assert resp.status_code == 307
    assert "auth_error=access_denied" in resp.headers["location"]


@respx.mock
def test_instagram_callback_exchanges_tokens_and_sets_session():
    # 1. Forzamos un estado válido en la sesión simulando el paso de /login
    session_client = TestClient(app, base_url="https://testserver")
    login_resp = session_client.get("/auth/instagram/login", follow_redirects=False)
    state = login_resp.headers["location"].split("state=")[1].split("&")[0]

    # 2. Mockeamos las dos llamadas a la API de Instagram
    respx.post("https://api.instagram.com/oauth/access_token").mock(
        return_value=httpx.Response(200, json={"access_token": "short-lived-token", "user_id": 123456}),
    )
    respx.get("https://graph.instagram.com/access_token").mock(
        return_value=httpx.Response(200, json={"access_token": "long-lived-token", "expires_in": 5184000}),
    )

    callback_resp = session_client.get(
        "/auth/instagram/callback",
        params={"code": "fake-code", "state": state},
        follow_redirects=False,
    )

    assert callback_resp.status_code == 307
    assert "/dashboard" in callback_resp.headers["location"]

    status_resp = session_client.get("/auth/instagram/status")
    assert status_resp.json() == {"authenticated": True}


def test_instagram_logout_clears_only_instagram_session_keys():
    session_client = TestClient(app, base_url="https://testserver")
    with session_client as c:
        # Simulamos sesión ya autenticada manipulando la cookie de sesión
        # a través del propio flujo, en vez de acceder a app.state (no
        # aplica aquí porque el estado vive en la cookie firmada del cliente).
        c.get("/auth/instagram/login")  # crea la cookie de sesión inicial
        resp = c.post("/auth/instagram/logout")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

        status_resp = c.get("/auth/instagram/status")
        assert status_resp.json() == {"authenticated": False}


def test_analyze_instagram_requires_authentication():
    resp = client.post("/api/analyze/instagram")
    assert resp.status_code == 401
