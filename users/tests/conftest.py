"""
Fixtures compartidas de pytest.

Nota: en este sandbox de desarrollo no siempre hay red disponible para
descargar los modelos completos de spaCy (en_core_web_sm / es_core_news_sm).
Por eso el fixture `patch_spacy_model` sustituye el cargador por un pipeline
ligero (`spacy.blank` + sentencizer) que ejerce exactamente la misma lógica
de negocio sin depender de los pesos del modelo descargado. En un entorno
con los modelos instalados (ver Dockerfile / README), el código de
producción usa el modelo completo sin ningún cambio.
"""
import os

import pytest
import spacy

os.environ.setdefault("REDDIT_CLIENT_ID", "test-client-id")
os.environ.setdefault("REDDIT_CLIENT_SECRET", "test-client-secret")
os.environ.setdefault("REDDIT_REDIRECT_URI", "http://localhost:3000/auth/reddit/callback")
os.environ.setdefault("REDDIT_USER_AGENT", "tfg-identity-exposure-tool-tests/0.1")
os.environ.setdefault("SESSION_SECRET_KEY", "test-secret-key")
os.environ.setdefault("FRONTEND_ORIGIN", "http://localhost:5173")


@pytest.fixture
def patch_spacy_model(monkeypatch):
    from app.nlp import fingerprint as fingerprint_module

    def _fake_get_model(lang: str):
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp

    monkeypatch.setattr(fingerprint_module, "_get_spacy_model", _fake_get_model)
