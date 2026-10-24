"""
Tests de app/skip_control.py y de su uso en el endpoint de streaming:
`POST /api/analyze/{platform}/skip` ("Saltar el resto del análisis") y el
paso de `skip` por `_build_report` hasta las dos pistas largas.
"""
import asyncio
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import analysis_router, skip_control
from app.config import settings
from app.main import app
from app.models.schemas import SocialPost, SocialProfile
from app.osint import username_correlation
from app.skip_control import SkipControl, cancel_pending, wait_or_skip

client = TestClient(app, base_url="https://testserver")


def _posts(n: int = 5) -> list[SocialPost]:
    return [
        SocialPost(
            id=f"p{i}",
            platform="reddit",
            type="post",
            group="madrid",
            tags=["madrid"],
            text=f"Post de prueba numero {i}",
            created_utc=datetime(2025, 1, 1, hour=i % 24, tzinfo=timezone.utc),
            score=1,
            permalink=f"https://reddit.com/r/test/{i}",
        )
        for i in range(n)
    ]


class TestSkipControl:
    @pytest.mark.asyncio
    async def test_starts_not_skipped_and_flips_once_requested(self):
        control = SkipControl()
        assert control.skipped is False

        control.request_skip()

        assert control.skipped is True
        await asyncio.wait_for(control.wait(), timeout=1)  # ya activado: no se queda esperando


class TestWaitOrSkip:
    @pytest.mark.asyncio
    async def test_returns_true_when_task_finishes_first(self):
        task = asyncio.ensure_future(asyncio.sleep(0, result="ok"))

        assert await wait_or_skip(task, SkipControl()) is True
        assert task.result() == "ok"

    @pytest.mark.asyncio
    async def test_returns_false_when_skip_fires_first_and_leaves_the_task_running(self):
        control = SkipControl()
        task = asyncio.ensure_future(asyncio.sleep(60))
        asyncio.get_running_loop().call_soon(control.request_skip)

        assert await wait_or_skip(task, control) is False
        assert not task.done()  # decidir si cancelarla es cosa de quien llama

        await cancel_pending([task])

    @pytest.mark.asyncio
    async def test_without_skip_just_waits_for_the_task(self):
        task = asyncio.ensure_future(asyncio.sleep(0, result=1))

        assert await wait_or_skip(task, None) is True

    @pytest.mark.asyncio
    async def test_does_not_reraise_the_task_exception(self):
        async def _boom():
            raise RuntimeError("fallo")

        task = asyncio.ensure_future(_boom())

        assert await wait_or_skip(task, SkipControl()) is True
        with pytest.raises(RuntimeError):
            task.result()


class TestCancelPending:
    @pytest.mark.asyncio
    async def test_cancels_only_unfinished_tasks(self):
        done_task = asyncio.ensure_future(asyncio.sleep(0, result="listo"))
        await done_task
        slow_task = asyncio.ensure_future(asyncio.sleep(60))

        await cancel_pending([done_task, slow_task])

        assert done_task.result() == "listo"
        assert slow_task.cancelled()


class TestRunRegistry:
    def test_request_skip_for_unknown_key_returns_false(self):
        assert skip_control.request_skip_for("reddit:nadie") is False

    def test_request_skip_for_registered_key_activates_it(self):
        control = SkipControl()
        skip_control.register_run("reddit:abc", control)
        try:
            assert skip_control.request_skip_for("reddit:abc") is True
            assert control.skipped is True
        finally:
            skip_control.unregister_run("reddit:abc", control)

    def test_unregister_does_not_remove_a_newer_run_with_the_same_key(self):
        old, new = SkipControl(), SkipControl()
        skip_control.register_run("reddit:abc", old)
        skip_control.register_run("reddit:abc", new)  # el EventSource reconectó
        try:
            skip_control.unregister_run("reddit:abc", old)

            assert skip_control.request_skip_for("reddit:abc") is True
            assert new.skipped is True
            assert old.skipped is False
        finally:
            skip_control.unregister_run("reddit:abc", new)


class TestSkipEndpoint:
    def test_unsupported_platform_returns_404(self):
        assert client.post("/api/analyze/tiktok/skip").status_code == 404

    def test_missing_session_returns_401(self):
        assert client.post("/api/analyze/reddit/skip").status_code == 401

    def test_no_running_analysis_returns_skipped_false(self, monkeypatch):
        monkeypatch.setitem(analysis_router._PLATFORM_CLIENT_FACTORIES, "reddit", lambda request: object())

        resp = client.post("/api/analyze/reddit/skip")

        assert resp.status_code == 200
        assert resp.json() == {"skipped": False}

    def test_running_analysis_is_skipped(self, monkeypatch):
        monkeypatch.setitem(analysis_router._PLATFORM_CLIENT_FACTORIES, "reddit", lambda request: object())
        control = SkipControl()
        # Sin sesión real, la clave de la ruta es "reddit:None" (ver _run_key).
        skip_control.register_run("reddit:None", control)
        try:
            resp = client.post("/api/analyze/reddit/skip")
        finally:
            skip_control.unregister_run("reddit:None", control)

        assert resp.json() == {"skipped": True}
        assert control.skipped is True


class TestBuildReportWithSkip:
    @pytest.mark.asyncio
    async def test_skip_reaches_the_correlation_task_and_report_is_still_generated(
        self, monkeypatch, patch_spacy_model
    ):
        """Al saltar, la comprobación de cuentas devuelve lo que llevara y el
        pipeline sigue hasta generar el informe completo (no se aborta)."""
        received = {}

        async def _fake_check(username, progress_callback=None, skip=None):
            received["skip"] = skip
            await skip.wait()  # "barrido largo" que solo acaba al saltar
            return [username_correlation.UsernameSiteResult(site="GitHub", url="https://github.com/u", exists=True)]

        monkeypatch.setattr(username_correlation, "check_username_across_sites", _fake_check)
        monkeypatch.setattr(settings, "enable_username_correlation", True)
        control = SkipControl()
        asyncio.get_running_loop().call_later(0.05, control.request_skip)

        profile = SocialProfile(platform="reddit", username="fake_user", posts=_posts())
        report = await asyncio.wait_for(analysis_router._build_report(profile, skip=control), timeout=30)

        assert received["skip"] is control
        assert report.related_accounts is not None
        assert [m.site for m in report.related_accounts.matches] == ["GitHub"]
        assert report.privacy_score is not None  # el resto del informe sí se generó

    @pytest.mark.asyncio
    async def test_without_skip_the_correlation_checker_is_called_exactly_as_before(
        self, monkeypatch, patch_spacy_model
    ):
        async def _fake_check(username, progress_callback=None):  # sin parámetro `skip`
            return []

        monkeypatch.setattr(username_correlation, "check_username_across_sites", _fake_check)
        monkeypatch.setattr(settings, "enable_username_correlation", True)

        profile = SocialProfile(platform="reddit", username="fake_user", posts=_posts())

        report = await analysis_router._build_report(profile)

        assert report.related_accounts is not None
