"""
Señales del texto agregadas por publicación y convertidas en
`InferredAttribute`. Dos puntos de entrada, ambos gratuitos y deterministas:

- `infer_lifestyle_attributes`: estilo de vida (aficiones, mascotas,
  vivienda, hábitos, salud, ideología...), bancos de `lifestyle_patterns.py`.
- `infer_exposure_attributes`: EXPOSICIÓN directa -- identificadores
  publicados (email, teléfono, DNI, IBAN, credenciales... enmascarados,
  `identifier_patterns.py`) y contexto (ubicación detallada, viajes y
  ausencias de casa, terceros y menores, relación de pareja,
  `context_patterns.py`).

Se llaman desde `report/generator.py` DESPUÉS de calcular el score: no
alteran `compute_score` ni el k-anonimato (no tienen tabla INE), solo
enriquecen la lista de atributos inferidos que ve el usuario y alimentan las
recomendaciones. Funcionan igual sin IA configurada, y se SUMAN a las
inferencias blandas de la IA (`ai_attribute_extraction.py`).
"""
from collections import defaultdict
from typing import Callable

from app.models.schemas import InferredAttribute, SocialPost
from app.nlp.context_patterns import detectar_contexto
from app.nlp.identifier_patterns import detectar_identificadores
from app.nlp.lifestyle_patterns import Hallazgo, detectar_estilo_de_vida

# Cada publicación adicional que repite la misma señal sube la confianza,
# hasta un tope: una señal blanda repetida nunca pasa a ser un hecho.
_BONUS_POR_REPETICION = 0.05
_CONFIANZA_MAXIMA = 0.9
_MAX_EVIDENCIAS = 5


def _agregar(posts: list[SocialPost], detector: Callable[[SocialPost], list[Hallazgo]]) -> list[InferredAttribute]:
    por_senal: dict[tuple[str, str], list[tuple[Hallazgo, str]]] = defaultdict(list)
    for post in posts:
        if not post.text and not post.tags:
            continue
        for hallazgo in detector(post):
            por_senal[(hallazgo.categoria, hallazgo.valor)].append((hallazgo, post.permalink))

    atributos: list[InferredAttribute] = []
    for (categoria, valor), apariciones in por_senal.items():
        base = max(h.confianza for h, _ in apariciones)
        confianza = min(base + _BONUS_POR_REPETICION * (len(apariciones) - 1), _CONFIANZA_MAXIMA)
        evidencias = list(dict.fromkeys(permalink for _, permalink in apariciones))[:_MAX_EVIDENCIAS]
        atributos.append(
            InferredAttribute(category=categoria, value=valor, confidence=round(confianza, 2), evidence=evidencias)
        )
    atributos.sort(key=lambda a: (-a.confidence, a.category, a.value))
    return atributos


def infer_lifestyle_attributes(posts: list[SocialPost]) -> list[InferredAttribute]:
    return _agregar(posts, lambda p: detectar_estilo_de_vida(p.text or "", p.tags))


def infer_exposure_attributes(posts: list[SocialPost]) -> list[InferredAttribute]:
    return _agregar(posts, lambda p: [*detectar_identificadores(p.text or ""), *detectar_contexto(p.text or "")])
