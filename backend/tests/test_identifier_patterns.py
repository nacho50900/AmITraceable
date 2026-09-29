"""Tests de los identificadores directos (app/nlp/identifier_patterns.py)."""
import pytest

from app.nlp.identifier_patterns import detectar_identificadores, enmascarar

# (texto, categoria, fragmento esperado en el valor, dato que NO debe aparecer sin enmascarar)
POSITIVOS = [
    ("Escríbeme a nacho.dev@uniovi.es", "contacto_publicado", "n***@uniovi.es", "nacho.dev"),
    ("llama al 612 345 678", "contacto_publicado", "Teléfono publicado", "612 345"),
    ("+34 612345678 por WhatsApp", "contacto_publicado", "Teléfono publicado", "612345"),
    ("fijo 91 234 56 78", "contacto_publicado", "Teléfono publicado", "91 234"),
    ("UK: +44 20 7946 0958", "contacto_publicado", "internacional", "7946"),
    ("Mi DNI es 12345678Z", "documento_identidad", "DNI publicado", "12345678"),
    ("NIE: X1234567L", "documento_identidad", "NIE publicado", "1234567"),
    ("CIF A58818501", "documento_identidad", "CIF publicado", "5881850"),
    ("Mi pasaporte: PAB123456", "documento_identidad", "pasaporte", "PAB123"),
    ("NUSS 281234567840", "documento_identidad", "Seguridad Social", "28123456"),
    ("el coche 1234 BCD aparcado", "documento_identidad", "Matrícula", "1234"),
    ("ref 9872023VH5797S0001WX", "documento_identidad", "catastral", "9872023"),
    ("IBAN ES91 2100 0418 4502 0005 1332", "dato_financiero", "IBAN publicado", "2100 0418"),
    ("cuenta 2100 0418 45 0200051332", "dato_financiero", "CCC", "2100 0418"),
    ("tarjeta 4111 1111 1111 1111", "dato_financiero", "tarjeta", "4111 1111"),
    ("BIC: CAIXESBBXXX", "dato_financiero", "BIC", "CAIXES"),
    ("wallet 0x52908400098527886E0F7030069857D2E4169EE7", "dato_financiero", "Ethereum", "5290840009"),
    ("desde 192.168.1.20 conecté", "identificador_tecnico", "IP", "192.168"),
    ("mac 00:1A:2B:3C:4D:5E", "identificador_tecnico", "MAC", "00:1A"),
    ("IMEI: 490154203237518", "identificador_tecnico", "IMEI", "4901542"),
    ("vivo en Calle Mayor 12, 3º B", "ubicacion_detallada", "Dirección postal", "Mayor 12"),
    ("CP 33003 Oviedo", "ubicacion_detallada", "33***", "33003"),
    ("estoy en 43.361914, -5.849389", "ubicacion_detallada", "43.36, -5.85", "361914"),
    ("sígueme @nacho_dev", "cuenta_externa", "@nacho_dev", None),
    ("insta: nacho.dev", "cuenta_externa", "Usuario de otra plataforma", None),
    ("https://www.instagram.com/nacho.dev/", "cuenta_externa", "instagram.com/nacho.dev", None),
    ("AKIAIOSFODNN7EXAMPLE", "credencial", "AWS", "AKIAIOSFODNN7"),
    ("ghp_" + "a" * 36, "credencial", "GitHub", "aaaaaaaa"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijk", "credencial", "JWT", "eyJhbGci"),
    ("-----BEGIN RSA PRIVATE KEY-----", "credencial", "clave privada", "BEGIN"),
    ("mongodb://root:secreto@localhost:27017/db", "credencial", "usuario y contraseña", "secreto"),
    ("mi contraseña es hunter2", "credencial", "Contraseña escrita", "hunter2"),
]


@pytest.mark.parametrize("texto,categoria,fragmento,secreto", POSITIVOS)
def test_detecta_y_enmascara(texto, categoria, fragmento, secreto):
    hallazgos = detectar_identificadores(texto)
    coincidentes = [h for h in hallazgos if h.categoria == categoria and fragmento in h.valor]
    assert coincidentes, f"{texto!r} -> {[(h.categoria, h.valor) for h in hallazgos]}"
    if secreto:
        assert all(secreto not in h.valor for h in hallazgos), "el valor sin enmascarar llegó al informe"


NEGATIVOS = [
    ("12345678A", "documento_identidad"),  # letra de control incorrecta
    ("X1234567A", "documento_identidad"),
    ("A58818502", "documento_identidad"),
    ("ES00 2100 0418 4502 0005 1332", "dato_financiero"),
    ("2100 0418 99 0200051332", "dato_financiero"),
    ("4111 1111 1111 1112", "dato_financiero"),
    ("quedamos a las 12:30:45", "identificador_tecnico"),
    ("versión 1.2.3.4.5", "identificador_tecnico"),
    ("IMEI: 490154203237519", "identificador_tecnico"),
    ("valor 95.123456, 10.123456", "ubicacion_detallada"),
    ("pedido 1234567890123", "contacto_publicado"),
    ("AKIAshort", "credencial"),
    ("Hoy hace buen día en Oviedo", "contacto_publicado"),
]


@pytest.mark.parametrize("texto,categoria", NEGATIVOS)
def test_no_detecta(texto, categoria):
    assert not [h for h in detectar_identificadores(texto) if h.categoria == categoria]


def test_credencial_no_reproduce_el_valor():
    (h,) = [h for h in detectar_identificadores("password: correcthorsebattery") if h.categoria == "credencial"]
    assert "correcthorse" not in h.valor


def test_validados_tienen_mas_confianza_que_solo_formato():
    dni = [h for h in detectar_identificadores("12345678Z") if h.categoria == "documento_identidad"][0]
    cp = [h for h in detectar_identificadores("CP 33003") if h.categoria == "ubicacion_detallada"][0]
    assert dni.confianza > cp.confianza


def test_email_no_genera_cuenta_externa():
    assert not [h for h in detectar_identificadores("a@b.com") if h.categoria == "cuenta_externa"]


@pytest.mark.parametrize(
    "etiqueta,valor,esperado",
    [
        ("email", "nacho@uniovi.es", "n***@uniovi.es"),
        ("iban", "ES91 2100 0418 4502 0005 1332", "**** **** **** **** **** 1332"),
        ("dni", "12345678Z", "******78Z"),
        ("codigo_postal", "33003", "33***"),
        ("direccion", "Calle Mayor 12", "Calle Mayor **"),
        ("coordenadas", "43.361914, -5.849389", "43.36, -5.85"),
    ],
)
def test_enmascarar(etiqueta, valor, esperado):
    assert enmascarar(etiqueta, valor) == esperado
