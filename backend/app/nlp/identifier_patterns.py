"""
Bancos de regex para IDENTIFICADORES DIRECTOS publicados en el texto del
propio usuario (captions, comentarios, bio): datos que identifican o
contactan con la persona sin necesidad de inferir nada -- email, teléfono,
DNI/NIE, IBAN, tarjeta, dirección postal, credenciales, enlaces a otras
cuentas...

Diferencias de diseño con `lifestyle_patterns.py` / `demographic_patterns.py`
(que buscan señales blandas con frase-ancla en primera persona):

- Aquí NO hay ancla: un DNI o un IBAN es un dato sensible aparezca donde
  aparezca. En su lugar, los patrones con dígito de control se VALIDAN
  (`validar_dni`, `validar_iban`, `validar_luhn`...) y lo que no valida se
  descarta: una secuencia numérica cualquiera es ruido, no un dato.
- El valor que llega al informe va ENMASCARADO (`enmascarar`): el informe
  se descarga en JSON y se manda a la IA de análisis, y no tiene sentido que
  un aviso de "has publicado tu IBAN" reproduzca el IBAN completo. Las
  credenciales ni siquiera incluyen el valor: solo su tipo.
- Confianza alta si el dato pasó su validación (`CONF_VALIDADO`); menor si
  solo encaja el formato (`CONF_FORMATO`) o es un enlace (`CONF_ENLACE`).

Cada entrada es un `Patron` con todo lo suyo en un sitio: etiqueta, regex,
validador, descripción para el informe y función de enmascarado. Si la regex
tiene un grupo de captura, el dato es ese grupo; si no, la coincidencia
completa.
"""
import ipaddress
import re
from collections.abc import Iterator
from typing import Callable, NamedTuple, Optional

from app.nlp.lifestyle_patterns import Hallazgo

CONF_VALIDADO = 0.9
CONF_FORMATO = 0.7
CONF_ENLACE = 0.5

Validador = Optional[Callable[[str], bool]]

# ---------------------------------------------------------------------------
# Validadores (dígitos de control)
# ---------------------------------------------------------------------------

_LETRAS_DNI = "TRWAGMYFPDXBNJZSQVHLCKE"
_PESOS_CCC = (1, 2, 4, 8, 5, 10, 9, 7, 3, 6)


def _alnum(texto: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", texto).upper()


def validar_dni(texto: str) -> bool:
    s = _alnum(texto)
    return len(s) == 9 and s[:8].isdigit() and _LETRAS_DNI[int(s[:8]) % 23] == s[8]


def validar_nie(texto: str) -> bool:
    s = _alnum(texto)
    if len(s) != 9 or s[0] not in "XYZ":
        return False
    numero = str("XYZ".index(s[0])) + s[1:8]
    return numero.isdigit() and _LETRAS_DNI[int(numero) % 23] == s[8]


def validar_cif(texto: str) -> bool:
    s = _alnum(texto)
    if not re.fullmatch(r"[ABCDEFGHJNPQRSUVW]\d{7}[0-9A-J]", s):
        return False
    d = s[1:8]
    pares = sum(int(d[i]) for i in (1, 3, 5))
    impares = sum(sum(divmod(int(d[i]) * 2, 10)) for i in (0, 2, 4, 6))
    control = (10 - (pares + impares) % 10) % 10
    letra = "JABCDEFGHI"[control]
    if s[0] in "PQRSNW":
        return s[8] == letra
    if s[0] in "ABEH":
        return s[8] == str(control)
    return s[8] in (str(control), letra)


def _luhn(digitos: str) -> bool:
    total = 0
    for i, c in enumerate(reversed(digitos)):
        d = int(c)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def validar_tarjeta(texto: str) -> bool:
    digitos = re.sub(r"\D", "", texto)
    return 13 <= len(digitos) <= 19 and _luhn(digitos)


def validar_imei(texto: str) -> bool:
    digitos = re.sub(r"\D", "", texto)
    return len(digitos) == 15 and _luhn(digitos)


def validar_iban(texto: str) -> bool:
    s = _alnum(texto)
    if not 15 <= len(s) <= 34 or not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", s):
        return False
    if s.startswith("ES") and len(s) != 24:
        return False
    reordenado = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in reordenado)) % 97 == 1


def _digito_control(cifras: str) -> int:
    resto = 11 - sum(int(c) * p for c, p in zip(cifras, _PESOS_CCC)) % 11
    return {10: 1, 11: 0}.get(resto, resto)


def validar_ccc(texto: str) -> bool:
    s = re.sub(r"\D", "", texto)
    if len(s) != 20:
        return False
    return _digito_control("00" + s[:8]) == int(s[8]) and _digito_control(s[10:]) == int(s[9])


def validar_ip(texto: str) -> bool:
    try:
        ipaddress.ip_address(texto)
    except ValueError:
        return False
    return True


def validar_coordenadas(texto: str) -> bool:
    try:
        lat, lon = (float(x) for x in re.split(r"\s*,\s*", texto.strip()))
    except ValueError:
        return False
    return abs(lat) <= 90 and abs(lon) <= 180


# ---------------------------------------------------------------------------
# Enmascarado
# ---------------------------------------------------------------------------


def _enmascarar_alnum(valor: str, visibles: int) -> str:
    """Sustituye por '*' todos los caracteres alfanuméricos salvo los
    últimos `visibles`; separadores y espacios se conservan."""
    total = sum(c.isalnum() for c in valor)
    vistos = 0
    resultado = []
    for c in valor:
        if c.isalnum():
            vistos += 1
            resultado.append(c if vistos > total - visibles else "*")
        else:
            resultado.append(c)
    return "".join(resultado)


def _mascara_email(valor: str) -> str:
    local, _, dominio = valor.partition("@")
    return f"{local[:1]}***@{dominio}"


def _mascara_coordenadas(valor: str) -> str:
    lat, lon = (float(x) for x in re.split(r"\s*,\s*", valor.strip()))
    return f"{lat:.2f}, {lon:.2f}"


def _mascara_ultimos(visibles: int) -> Callable[[str], str]:
    return lambda valor: _enmascarar_alnum(valor, visibles)


_MASCARA_LARGA = _mascara_ultimos(4)  # cuentas, tarjetas, MAC, IMEI...
_MASCARA_DOCUMENTO = _mascara_ultimos(3)  # DNI, pasaporte, matrícula...
_MASCARA_CORTA = _mascara_ultimos(2)


def _sin_mascara(valor: str) -> str:
    return valor


class Patron(NamedTuple):
    etiqueta: str
    regex: str
    descripcion: str
    mascara: Callable[[str], str]
    validador: Validador = None


# ---------------------------------------------------------------------------
# Bancos
# ---------------------------------------------------------------------------

_CONTACTO_PUBLICADO = "contacto_publicado"
_DOCUMENTO_IDENTIDAD = "documento_identidad"
_DATO_FINANCIERO = "dato_financiero"
_IDENTIFICADOR_TECNICO = "identificador_tecnico"
_UBICACION_DETALLADA = "ubicacion_detallada"
_CUENTA_EXTERNA = "cuenta_externa"
_CREDENCIAL = "credencial"

_DESC_MATRICULA = "Matrícula de vehículo publicada"
_DESC_IP = "Dirección IP publicada"

CONTACTO: tuple[Patron, ...] = (
    Patron(
        "email", r"(?<![\w.%+-])[\w.%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])",
        "Correo electrónico publicado", _mascara_email,
    ),
    Patron(
        "telefono_es", r"(?<!\d)(?<!\d[\s.-])(?<![\w+])(?:(?:\+|00)34[\s.-]?)?[6-9](?:[\s.-]?\d){8}(?![\s.-]?\d)",
        "Teléfono publicado", _MASCARA_CORTA,
    ),
    Patron(
        "telefono_internacional", r"(?<![\w+])\+(?!34)\d{1,3}(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?!\d)",
        "Teléfono internacional publicado", _MASCARA_CORTA,
    ),
)

DOCUMENTOS: tuple[Patron, ...] = (
    Patron("dni", r"(?i)(?<![\w])\d{8}[\s-]?[A-HJ-NP-TV-Z](?![\w])", "DNI publicado (letra de control válida)", _MASCARA_DOCUMENTO, validar_dni),
    Patron("nie", r"(?i)(?<![\w])[XYZ][\s.-]?\d{7}[\s-]?[A-Z](?![\w])", "NIE publicado (letra de control válida)", _MASCARA_DOCUMENTO, validar_nie),
    Patron(
        "cif", r"(?i)(?<![\w])[ABCDEFGHJNPQRSUVW][\s-]?\d{7}[\s-]?[0-9A-J](?![\w])",
        "CIF publicado (dígito de control válido)", _MASCARA_DOCUMENTO, validar_cif,
    ),
    Patron(
        "pasaporte", r"(?i)\b(?:pasaporte|passport)\b\D{0,12}?([A-Z]{2,3}\d{6,7}[A-Z]?|[A-Z]\d{8})\b",
        "Número de pasaporte publicado", _MASCARA_DOCUMENTO,
    ),
    Patron(
        "seguridad_social",
        r"(?i)(?:seguridad social|n\.?\s?s\.?\s?s\.?|nuss|afiliaci[oó]n)\D{0,15}?(\d{2}[\s/-]?\d{8}[\s/-]?\d{2})(?!\d)",
        "Número de la Seguridad Social publicado", _MASCARA_DOCUMENTO,
    ),
    Patron("matricula", r"(?i)(?<![\w])\d{4}[\s-]?[BCDFGHJKLMNPRSTVWXYZ]{3}(?![\w])", _DESC_MATRICULA, _MASCARA_DOCUMENTO),
    Patron(
        "matricula_antigua", r"(?i)\bmatr[ií]cula\b\D{0,12}?([A-Z]{1,2}[\s-]?\d{4}[\s-]?[A-Z]{1,2})\b",
        _DESC_MATRICULA, _MASCARA_DOCUMENTO,
    ),
    Patron(
        "referencia_catastral", r"(?i)(?<![\w])\d{7}[A-Z]{2}\d{4}[A-Z]\d{4}[A-Z]{2}(?![\w])",
        "Referencia catastral de un inmueble publicada", _MASCARA_DOCUMENTO,
    ),
)

FINANCIERO: tuple[Patron, ...] = (
    Patron(
        "iban", r"(?i)(?<![\w])[A-Z]{2}\d{2}(?:[\s-]?[A-Z0-9]{4}){2,7}(?:[\s-]?[A-Z0-9]{1,3})?(?![\w])",
        "IBAN publicado (dígitos de control válidos)", _MASCARA_LARGA, validar_iban,
    ),
    Patron("ccc", r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{2}[\s-]?\d{10}(?!\d)", "Cuenta bancaria (CCC) publicada", _MASCARA_LARGA, validar_ccc),
    Patron(
        "tarjeta", r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
        "Número de tarjeta publicado (pasa el algoritmo de Luhn)", _MASCARA_LARGA, validar_tarjeta,
    ),
    Patron(
        "bic_swift", r"(?i)\b(?:bic|swift)\b\s*[:#]?\s*([A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b",
        "Código BIC/SWIFT publicado", _MASCARA_CORTA,
    ),
    Patron(
        "cripto_btc", r"(?<![\w])(?:bc1[a-z0-9]{25,59}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})(?![\w])",
        "Dirección de monedero Bitcoin publicada", _MASCARA_CORTA,
    ),
    Patron("cripto_eth", r"(?<![\w])0x[a-fA-F0-9]{40}(?![\w])", "Dirección de monedero Ethereum publicada", _MASCARA_CORTA),
)

TECNICO: tuple[Patron, ...] = (
    Patron(
        "ipv4", r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])",
        _DESC_IP, _MASCARA_CORTA,
    ),
    Patron("ipv6", r"(?<![\w:])(?:[A-Fa-f0-9]{0,4}:){2,7}[A-Fa-f0-9]{0,4}(?![\w:])", _DESC_IP, _MASCARA_LARGA, validar_ip),
    Patron("mac", r"(?<![\w:-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![\w:-])", "Dirección MAC publicada", _MASCARA_LARGA),
    Patron("imei", r"(?i)\bimei\b\D{0,6}?(\d{15})(?!\d)", "IMEI de un móvil publicado", _MASCARA_LARGA, validar_imei),
)

DIRECCION: tuple[Patron, ...] = (
    Patron(
        "direccion",
        r"(?i)\b(?:c/\s*|(?:calle|avda?\.?|avenida|plaza|pza\.?|paseo|camino|carretera|ctra\.?|travesía|glorieta|"
        r"ronda|urbanización|polígono)\s+)[^\d,\n]{2,50}?,?\s*(?:n[ºo°.]?\s*)?\d{1,3}"
        r"(?:\s*,?\s*\d{1,2}[ºª°]?\s*[A-Za-z]?\b)?",
        "Dirección postal publicada", lambda valor: re.sub(r"\d", "*", valor),
    ),
    Patron(
        "codigo_postal", r"(?i)(?:\bc\.?\s?p\.?|c[oó]digo postal)\s*[:.]?\s*((?:0[1-9]|[1-4]\d|5[0-2])\d{3})(?!\d)",
        "Código postal publicado", lambda valor: valor[:2] + "***",
    ),
    Patron(
        "coordenadas", r"(?<![\d.])[-+]?\d{1,3}\.\d{4,}\s*,\s*[-+]?\d{1,3}\.\d{4,}(?![\d])",
        "Coordenadas GPS publicadas (redondeadas a ~1 km)", _mascara_coordenadas, validar_coordenadas,
    ),
)

# Los enlaces y usuarios públicos se muestran tal cual (ya son públicos).
CUENTAS_EXTERNAS: tuple[Patron, ...] = (
    Patron("handle", r"(?<![\w@./])@[A-Za-z0-9_.]{2,29}[A-Za-z0-9_]", "Nombre de usuario de otra cuenta", _sin_mascara),
    Patron(
        "usuario_con_plataforma",
        r"(?i)\b(?:insta(?:gram)?|ig|twitter|tiktok|snap(?:chat)?|telegram|discord|twitch|steam|linkedin|github)\b"
        r"\s*(?:[:=]\s*@?|@)[A-Za-z0-9_.]{2,29}[A-Za-z0-9_]",
        "Usuario de otra plataforma", _sin_mascara,
    ),
    Patron(
        "perfil_social",
        r"(?i)(?<![\w.-])(?:https?://)?(?:[\w-]+\.)?(?:instagram\.com|twitter\.com|x\.com|facebook\.com|fb\.com|"
        r"linkedin\.com|tiktok\.com|github\.com|gitlab\.com|youtube\.com|youtu\.be|twitch\.tv|t\.me|wa\.me|"
        r"snapchat\.com|threads\.net|reddit\.com|pinterest\.[a-z]{2,3}|strava\.com|spotify\.com|"
        r"steamcommunity\.com|discord\.gg|bsky\.app|vinted\.[a-z]{2,3}|wallapop\.com)/[^\s<>\"')]+",
        "Enlace a un perfil en otra plataforma", _sin_mascara,
    ),
)

# Las credenciales NUNCA reproducen su valor: solo el tipo (ver `_valor_informe`).
CREDENCIALES: tuple[Patron, ...] = (
    Patron("aws_access_key", r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "Posible clave de acceso de AWS publicada", _sin_mascara),
    Patron("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}\b", "Posible clave de API de Google publicada", _sin_mascara),
    Patron("github_token", r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b", "Posible token de GitHub publicado", _sin_mascara),
    Patron("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b", "Posible token de Slack publicado", _sin_mascara),
    Patron("stripe_key", r"\b[sr]k_live_[0-9A-Za-z]{16,}\b", "Posible clave secreta de Stripe publicada", _sin_mascara),
    Patron("api_key_llm", r"\bsk-(?:ant-|proj-)[A-Za-z0-9_-]{20,}\b", "Posible clave de API de un servicio de IA publicada", _sin_mascara),
    Patron("telegram_bot", r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b", "Posible token de bot de Telegram publicado", _sin_mascara),
    Patron("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b", "Posible token JWT publicado", _sin_mascara),
    Patron("clave_privada", r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY(?: BLOCK)?-----", "Posible clave privada publicada", _sin_mascara),
    Patron("url_con_credenciales", r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]+@[^\s/]+", "URL con usuario y contraseña incrustados", _sin_mascara),
    Patron("contrasena_declarada", r"(?i)\b(?:contrase[ñn]a|password|passwd|pwd|clave)\s*(?:es|:|=)\s*\S{4,}", "Contraseña escrita en el texto", _sin_mascara),
)

# (banco, categoría del atributo inferido)
BANCOS: tuple[tuple[tuple[Patron, ...], str], ...] = (
    (CONTACTO, _CONTACTO_PUBLICADO),
    (DOCUMENTOS, _DOCUMENTO_IDENTIDAD),
    (FINANCIERO, _DATO_FINANCIERO),
    (TECNICO, _IDENTIFICADOR_TECNICO),
    (DIRECCION, _UBICACION_DETALLADA),
    (CUENTAS_EXTERNAS, _CUENTA_EXTERNA),
    (CREDENCIALES, _CREDENCIAL),
)

_COMPILADOS = tuple(
    (tuple((patron, re.compile(patron.regex)) for patron in patrones), categoria) for patrones, categoria in BANCOS
)
_MASCARA_POR_ETIQUETA = {patron.etiqueta: patron.mascara for patrones, _ in BANCOS for patron in patrones}


def enmascarar(etiqueta: str, valor: str) -> str:
    return _MASCARA_POR_ETIQUETA[etiqueta](valor)


def _valor_informe(patron: Patron, categoria: str, dato: str) -> str:
    """Texto que llega al informe: las credenciales solo llevan su tipo; el
    resto, su descripción y el dato ya enmascarado."""
    if categoria == _CREDENCIAL:
        return patron.descripcion
    return f"{patron.descripcion}: {patron.mascara(dato)}"


def _confianza(patron: Patron, categoria: str) -> float:
    if categoria == _CUENTA_EXTERNA:
        return CONF_ENLACE
    return CONF_VALIDADO if patron.validador is not None else CONF_FORMATO


def _dato(patron: Patron, rx: re.Pattern[str], texto: str) -> Iterator[str]:
    """Datos que encajan con el patrón y pasan su validación."""
    for m in rx.finditer(texto):
        grupo = 1 if rx.groups and m.group(1) is not None else 0
        dato = m.group(grupo)
        if patron.validador is None or patron.validador(dato):
            yield dato


def detectar_identificadores(texto: str) -> list[Hallazgo]:
    """Identificadores directos de un texto, ya enmascarados. Sin
    duplicados (misma categoría y mismo valor enmascarado)."""
    hallazgos = [
        Hallazgo(categoria, _valor_informe(patron, categoria, dato), _confianza(patron, categoria))
        for patrones, categoria in _COMPILADOS
        for patron, rx in patrones
        for dato in _dato(patron, rx, texto)
    ]
    return list(dict.fromkeys(hallazgos))
