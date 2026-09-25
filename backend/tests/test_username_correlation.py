"""
Tests de app/osint/username_correlation.py: capa fina sobre la libreria
`maigret`, comprobacion de EXISTENCIA unicamente (ver docstring del modulo
para por que is_parsing_enabled=False / is_enrich_enabled=False bastan
para excluir tanto la extraccion de contenido como la recursion).

No se mockea HTTP de bajo nivel aqui (a diferencia de versiones anteriores
de este fichero): se mockea `_maigret_check`, el punto de entrada a la
propia libreria de Maigret, porque lo que este modulo controla y necesita
tests es (a) que llama a esa funcion con las flags correctas, y (b) que
traduce su resultado (MaigretCheckStatus) a nuestro UsernameSiteResult
correctamente -- el motor de deteccion HTTP en si ya es responsabilidad
(y esta testeado) por la propia libreria Maigret, no por este proyecto.
"""
import pytest
from fastapi.testclient import TestClient

from app import osint_router
from app.main import app
from app.osint.username_correlation import (
    UsernameSiteResult,
    check_username_across_sites,
    get_database,
)
from maigret.result import MaigretCheckResult, MaigretCheckStatus

client = TestClient(app)


def _fake_site_result(status: MaigretCheckStatus, url_user: str = "", url_main: str = ""):
    return {
        "status": MaigretCheckResult(
            username="comandante", site_name="Sitio", site_url_user=url_user, status=status
        ),
        "url_user": url_user,
        "url_main": url_main,
    }


class TestCheckUsernameAcrossSites:
    @pytest.mark.asyncio
    async def test_empty_username_returns_no_results_without_calling_maigret(self, monkeypatch):
        async def _should_not_be_called(**kwargs):
            raise AssertionError("no deberia llamarse a maigret con username vacio")

        monkeypatch.setattr(
            "app.osint.username_correlation._maigret_check", _should_not_be_called
        )

        results = await check_username_across_sites("   ")

        assert results == []

    @pytest.mark.asyncio
    async def test_claimed_status_maps_to_exists_true(self, monkeypatch):
        async def _fake_check(**kwargs):
            return {
                "GitHub": _fake_site_result(
                    MaigretCheckStatus.CLAIMED, url_user="https://github.com/comandante"
                )
            }

        monkeypatch.setattr("app.osint.username_correlation._maigret_check", _fake_check)

        results = await check_username_across_sites("comandante", sites={})

        assert results == [
            UsernameSiteResult(site="GitHub", url="https://github.com/comandante", exists=True)
        ]

    @pytest.mark.asyncio
    async def test_available_status_maps_to_exists_false(self, monkeypatch):
        async def _fake_check(**kwargs):
            return {
                "GitHub": _fake_site_result(
                    MaigretCheckStatus.AVAILABLE, url_user="https://github.com/comandante"
                )
            }

        monkeypatch.setattr("app.osint.username_correlation._maigret_check", _fake_check)

        results = await check_username_across_sites("comandante", sites={})

        assert results[0].exists is False

    @pytest.mark.asyncio
    async def test_unknown_status_maps_to_exists_none(self, monkeypatch):
        """UNKNOWN = Maigret no pudo determinarlo (timeout, bloqueo
        anti-bot, error del sitio) -- nunca se interpreta como "no
        existe"."""

        async def _fake_check(**kwargs):
            return {"GitHub": _fake_site_result(MaigretCheckStatus.UNKNOWN)}

        monkeypatch.setattr("app.osint.username_correlation._maigret_check", _fake_check)

        results = await check_username_across_sites("comandante", sites={})

        assert results[0].exists is None

    @pytest.mark.asyncio
    async def test_illegal_status_is_excluded_entirely(self, monkeypatch):
        """ILLEGAL = el username no cumple el formato de ese sitio
        (regexCheck) -- no es "no existe", el sitio no aplica y se omite
        del resultado."""

        async def _fake_check(**kwargs):
            return {
                "SitioEstricto": _fake_site_result(MaigretCheckStatus.ILLEGAL),
                "GitHub": _fake_site_result(
                    MaigretCheckStatus.CLAIMED, url_user="https://github.com/comandante"
                ),
            }

        monkeypatch.setattr("app.osint.username_correlation._maigret_check", _fake_check)

        results = await check_username_across_sites("comandante", sites={})

        assert [r.site for r in results] == ["GitHub"]

    @pytest.mark.asyncio
    async def test_url_falls_back_to_url_main_when_no_url_user(self, monkeypatch):
        async def _fake_check(**kwargs):
            return {
                "Sitio": _fake_site_result(
                    MaigretCheckStatus.CLAIMED, url_user="", url_main="https://sitio.test/"
                )
            }

        monkeypatch.setattr("app.osint.username_correlation._maigret_check", _fake_check)

        results = await check_username_across_sites("comandante", sites={})

        assert results[0].url == "https://sitio.test/"

    @pytest.mark.asyncio
    async def test_dossier_features_are_explicitly_disabled_at_the_call_site(self, monkeypatch):
        """El punto central de este modulo: cualquier llamada a Maigret
        pasa is_parsing_enabled=False e is_enrich_enabled=False, sin
        excepcion. Si algun dia alguien las quita al tocar este fichero,
        este test debe fallar."""
        seen_kwargs = {}

        async def _fake_check(**kwargs):
            seen_kwargs.update(kwargs)
            return {}

        monkeypatch.setattr("app.osint.username_correlation._maigret_check", _fake_check)

        await check_username_across_sites("comandante", sites={})

        assert seen_kwargs["is_parsing_enabled"] is False
        assert seen_kwargs["is_enrich_enabled"] is False

    @pytest.mark.asyncio
    async def test_default_sites_exclude_tor_i2p_dns(self, monkeypatch):
        seen_kwargs = {}

        async def _fake_check(**kwargs):
            seen_kwargs.update(kwargs)
            return {}

        monkeypatch.setattr("app.osint.username_correlation._maigret_check", _fake_check)

        await check_username_across_sites("comandante")  # sin `sites` -> usa la base completa

        site_dict = seen_kwargs["site_dict"]
        protocols = {getattr(site, "protocol", None) for site in site_dict.values()}
        assert not protocols & {"tor", "i2p", "dns"}


class TestProgressCallback:
    """`progress_callback` es opcional (ver docstring del modulo) -- estos
    tests no esperan `PROGRESS_POLL_INTERVAL_SECONDS` real (1s): con
    `_maigret_check` resolviendo casi al instante, la tarea de sondeo se
    cancela antes de disparar ningun evento intermedio, asi que lo que se
    prueba aqui es el evento FINAL de cierre (100%), que se emite siempre
    tras `await`, y que sin callback no hay ningun overhead ni error."""

    @pytest.mark.asyncio
    async def test_no_callback_means_no_progress_calls_and_no_error(self, monkeypatch):
        async def _fake_check(**kwargs):
            assert "query_notify" in kwargs  # se pasa igual, aunque no haya callback
            return {}

        monkeypatch.setattr("app.osint.username_correlation._maigret_check", _fake_check)

        results = await check_username_across_sites(
            "comandante", sites={"Sitio": object()}, progress_callback=None
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_emits_a_final_100_percent_event_after_completion(self):
        events = []

        async def _progress_callback(stage, counts):
            events.append((stage, counts))

        async def _fake_check(**kwargs):
            return {}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.osint.username_correlation._maigret_check", _fake_check)
            results = await check_username_across_sites(
                "comandante",
                sites={"A": object(), "B": object()},
                progress_callback=_progress_callback,
            )

        assert results == []
        assert events  # al menos el evento final
        stage, counts = events[-1]
        assert counts["track"] == "correlacion_cuentas"
        assert counts["accounts_checked"] == 2
        assert counts["total_accounts"] == 2

    @pytest.mark.asyncio
    async def test_empty_username_never_calls_the_progress_callback(self):
        async def _should_not_be_called(stage, counts):
            raise AssertionError("no deberia emitir progreso para un username vacio")

        results = await check_username_across_sites("   ", progress_callback=_should_not_be_called)

        assert results == []


class TestMaigretDatabase:
    """Lee el fichero de datos real que trae `maigret` instalado -- dato
    estatico del paquete, ninguna peticion de red."""

    def test_loads_a_large_number_of_sites(self):
        db = get_database()
        assert len(db.sites) > 3000

    def test_is_cached_across_calls(self):
        assert get_database() is get_database()


class TestUsernameCorrelationEndpoint:
    """Tests del endpoint HTTP, con el checker mockeado (los tests de
    arriba ya cubren la logica de mapeo de estados de Maigret)."""

    def test_returns_matches_from_the_checker(self, monkeypatch):
        async def _fake_check(username, sites=None, **kwargs):
            assert username == "comandante"
            return [
                UsernameSiteResult(site="GitHub", url="https://github.com/comandante", exists=True),
                UsernameSiteResult(site="GitLab", url="https://gitlab.com/comandante", exists=False),
            ]

        monkeypatch.setattr(osint_router, "check_username_across_sites", _fake_check)

        resp = client.post("/api/username-correlation", json={"username": "comandante"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == "comandante"
        assert body["matches"] == [
            {"site": "GitHub", "url": "https://github.com/comandante", "exists": True},
            {"site": "GitLab", "url": "https://gitlab.com/comandante", "exists": False},
        ]
        assert "checked_at" in body

    def test_empty_username_returns_422_without_calling_the_checker(self, monkeypatch):
        def _should_not_be_called(*args, **kwargs):
            raise AssertionError("no deberia llamarse al checker con username vacio")

        monkeypatch.setattr(osint_router, "check_username_across_sites", _should_not_be_called)

        resp = client.post("/api/username-correlation", json={"username": "   "})

        assert resp.status_code == 422

    def test_username_is_stripped_before_checking(self, monkeypatch):
        seen = {}

        async def _fake_check(username, sites=None, **kwargs):
            seen["username"] = username
            return []

        monkeypatch.setattr(osint_router, "check_username_across_sites", _fake_check)

        resp = client.post("/api/username-correlation", json={"username": "  comandante  "})

        assert resp.status_code == 200
        assert seen["username"] == "comandante"
        assert resp.json()["username"] == "comandante"
