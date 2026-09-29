"""
Señales de estilo de vida en el texto, agregadas por publicación y
convertidas en `InferredAttribute` (ver `lifestyle_patterns.py` para los
bancos de regex).

Se llaman desde `report/generator.py` DESPUÉS de calcular el score: no
alteran `compute_score` ni el k-anonimato (no tienen tabla INE), solo
enriquecen la lista de atributos inferidos que ve el usuario. Es un
complemento por regex, gratuito y determinista, de las inferencias blandas de
la IA (`ai_attribute_extraction.py`) -- funciona igual sin IA configurada, y
las dos listas se SUMAN.
"""
from collections import defaultdict

from app.models.schemas import InferredAttribute, SocialPost
from app.nlp.lifestyle_patterns import Hallazgo, detectar_estilo_de_vida

# Cada publicación adicional que repite la misma señal sube la confianza,
# hasta un tope: una señal blanda repetida nunca pasa a ser un hecho.
_BONUS_POR_REPETICION = 0.05
_CONFIANZA_MAXIMA = 0.9
_MAX_EVIDENCIAS = 5


def infer_lifestyle_attributes(posts: list[SocialPost]) -> list[InferredAttribute]:
    por_senal: dict[tuple[str, str], list[tuple[Hallazgo, str]]] = defaultdict(list)
    for post in posts:
        if not post.text and not post.tags:
            continue
        for hallazgo in detectar_estilo_de_vida(post.text or "", post.tags):
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
