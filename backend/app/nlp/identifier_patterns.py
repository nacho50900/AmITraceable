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

Cada entrada es `(etiqueta, patrón, validador | None)`. Si el patrón tiene un
grupo de captura, el dato es ese grupo; si no, la coincidencia completa.
"""
import ipaddress
import re
from typing import Callable, Optional

from app.nlp.lifestyle_patterns import Hallazgo

CONF_VALIDADO = 0.9
CONF_FORMATO = 0.7
CONF_ENLACE = 0.5

Validador = Optional[Callable[[str], bool]]
Entrada = tuple[str, str, Validador]

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
# Bancos
# ---------------------------------------------------------------------------

CONTACTO: tuple[Entrada, ...] = (
    ("email", r"(?<![\w.%+-])[\w.%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])", None),
    (
        "telefono_es",
        r"(?<!\d)(?<!\d[\s.-])(?<![\w+])(?:(?:\+|00)34[\s.-]?)?[6-9](?:[\s.-]?\d){8}(?![\s.-]?\d)",
        None,
    ),
    ("telefono_internacional", r"(?<![\w+])\+(?!34)\d{1,3}(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?!\d)", None),
)

DOCUMENTOS: tuple[Entrada, ...] = (
    ("dni", r"(?i)(?<![\w])\d{8}[\s-]?[A-HJ-NP-TV-Z](?![\w])", validar_dni),
    ("nie", r"(?i)(?<![\w])[XYZ][\s.-]?\d{7}[\s-]?[A-Z](?![\w])", validar_nie),
    ("cif", r"(?i)(?<![\w])[ABCDEFGHJNPQRSUVW][\s-]?\d{7}[\s-]?[0-9A-J](?![\w])", validar_cif),
    ("pasaporte", r"(?i)\b(?:pasaporte|passport)\b\D{0,12}?([A-Z]{2,3}\d{6,7}[A-Z]?|[A-Z]\d{8})\b", None),
    (
        "seguridad_social",
        r"(?i)(?:seguridad social|n\.?\s?s\.?\s?s\.?|nuss|afiliaci[oó]n)\D{0,15}?(\d{2}[\s/-]?\d{8}[\s/-]?\d{2})(?!\d)",
        None,
    ),
    ("matricula", r"(?i)(?<![\w])\d{4}[\s-]?[BCDFGHJKLMNPRSTVWXYZ]{3}(?![\w])", None),
    ("matricula_antigua", r"(?i)\bmatr[ií]cula\b\D{0,12}?([A-Z]{1,2}[\s-]?\d{4}[\s-]?[A-Z]{1,2})\b", None),
    ("referencia_catastral", r"(?i)(?<![\w])\d{7}[A-Z]{2}\d{4}[A-Z]\d{4}[A-Z]{2}(?![\w])", None),
)

FINANCIERO: tuple[Entrada, ...] = (
    ("iban", r"(?i)(?<![\w])[A-Z]{2}\d{2}(?:[\s-]?[A-Z0-9]{4}){2,7}(?:[\s-]?[A-Z0-9]{1,3})?(?![\w])", validar_iban),
    ("ccc", r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{2}[\s-]?\d{10}(?!\d)", validar_ccc),
    ("tarjeta", r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)", validar_tarjeta),
    ("bic_swift", r"(?i)\b(?:bic|swift)\b\s*[:#]?\s*([A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b", None),
    ("cripto_btc", r"(?<![\w])(?:bc1[a-z0-9]{25,59}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})(?![\w])", None),
    ("cripto_eth", r"(?<![\w])0x[a-fA-F0-9]{40}(?![\w])", None),
)

TECNICO: tuple[Entrada, ...] = (
    ("ipv4", r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])", None),
    ("ipv6", r"(?<![\w:])(?:[A-Fa-f0-9]{0,4}:){2,7}[A-Fa-f0-9]{0,4}(?![\w:])", validar_ip),
    ("mac", r"(?<![\w:-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![\w:-])", None),
    ("imei", r"(?i)\bimei\b\D{0,6}?(\d{15})(?!\d)", validar_imei),
)

DIRECCION: tuple[Entrada, ...] = (
    (
        "direccion",
        r"(?i)\b(?:c/\s*|(?:calle|avda?\.?|avenida|plaza|pza\.?|paseo|camino|carretera|ctra\.?|travesía|glorieta|"
        r"ronda|urbanización|polígono)\s+)[^\d,\n]{2,50}?,?\s*(?:n[ºo°.]?\s*)?\d{1,3}"
        r"(?:\s*,?\s*\d{1,2}[ºª°]?\s*[A-Za-z]?\b)?",
        None,
    ),
    (
        "codigo_postal",
        r"(?i)(?:\bc\.?\s?p\.?|c[oó]digo postal)\s*[:.]?\s*((?:0[1-9]|[1-4]\d|5[0-2])\d{3})(?!\d)",
        None,
    ),
    ("coordenadas", r"(?<![\d.])[-+]?\d{1,3}\.\d{4,}\s*,\s*[-+]?\d{1,3}\.\d{4,}(?![\d])", validar_coordenadas),
)

CUENTAS_EXTERNAS: tuple[Entrada, ...] = (
    ("handle", r"(?<![\w@./])@[A-Za-z0-9_.]{2,29}[A-Za-z0-9_]", None),
    (
        "usuario_con_plataforma",
        r"(?i)\b(?:insta(?:gram)?|ig|twitter|tiktok|snap(?:chat)?|telegram|discord|twitch|steam|linkedin|github)\b"
        r"\s*(?:[:=]\s*@?|@)[A-Za-z0-9_.]{2,29}[A-Za-z0-9_]",
        None,
    ),
    (
        "perfil_social",
        r"(?i)(?<![\w.-])(?:https?://)?(?:[\w-]+\.)?(?:instagram\.com|twitter\.com|x\.com|facebook\.com|fb\.com|"
        r"linkedin\.com|tiktok\.com|github\.com|gitlab\.com|youtube\.com|youtu\.be|twitch\.tv|t\.me|wa\.me|"
        r"snapchat\.com|threads\.net|reddit\.com|pinterest\.[a-z]{2,3}|strava\.com|spotify\.com|"
        r"steamcommunity\.com|discord\.gg|bsky\.app|vinted\.[a-z]{2,3}|wallapop\.com)/[^\s<>\"')]+",
        None,
    ),
)

CREDENCIALES: tuple[Entrada, ...] = (
    ("aws_access_key", r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", None),
    ("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}\b", None),
    ("github_token", r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b", None),
    ("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b", None),
    ("stripe_key", r"\b[sr]k_live_[0-9A-Za-z]{16,}\b", None),
    ("api_key_llm", r"\bsk-(?:ant-|proj-)[A-Za-z0-9_-]{20,}\b", None),
    ("telegram_bot", r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b", None),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b", None),
    ("clave_privada", r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY(?: BLOCK)?-----", None),
    ("url_con_credenciales", r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]+@[^\s/]+", None),
    ("contrasena_declarada", r"(?i)\b(?:contrase[ñn]a|password|passwd|pwd|clave)\s*(?:es|:|=)\s*\S{4,}", None),
)

# (banco, categoría del atributo inferido, descripción legible por etiqueta)
BANCOS: tuple[tuple[tuple[Entrada, ...], str, dict[str, str]], ...] = (
    (
        CONTACTO,
        "contacto_publicado",
        {
            "email": "Correo electrónico publicado",
            "telefono_es": "Teléfono publicado",
            "telefono_internacional": "Teléfono internacional publicado",
        },
    ),
    (
        DOCUMENTOS,
        "documento_identidad",
        {
            "dni": "DNI publicado (letra de control válida)",
            "nie": "NIE publicado (letra de control válida)",
            "cif": "CIF publicado (dígito de control válido)",
            "pasaporte": "Número de pasaporte publicado",
            "seguridad_social": "Número de la Seguridad Social publicado",
            "matricula": "Matrícula de vehículo publicada",
            "matricula_antigua": "Matrícula de vehículo publicada",
            "referencia_catastral": "Referencia catastral de un inmueble publicada",
        },
    ),
    (
        FINANCIERO,
        "dato_financiero",
        {
            "iban": "IBAN publicado (dígitos de control válidos)",
            "ccc": "Cuenta bancaria (CCC) publicada",
            "tarjeta": "Número de tarjeta publicado (pasa el algoritmo de Luhn)",
            "bic_swift": "Código BIC/SWIFT publicado",
            "cripto_btc": "Dirección de monedero Bitcoin publicada",
            "cripto_eth": "Dirección de monedero Ethereum publicada",
        },
    ),
    (
        TECNICO,
        "identificador_tecnico",
        {
            "ipv4": "Dirección IP publicada",
            "ipv6": "Dirección IP publicada",
            "mac": "Dirección MAC publicada",
            "imei": "IMEI de un móvil publicado",
        },
    ),
    (
        DIRECCION,
        "ubicacion_detallada",
        {
            "direccion": "Dirección postal publicada",
            "codigo_postal": "Código postal publicado",
            "coordenadas": "Coordenadas GPS publicadas (redondeadas a ~1 km)",
        },
    ),
    (
        CUENTAS_EXTERNAS,
        "cuenta_externa",
        {
            "handle": "Nombre de usuario de otra cuenta",
            "usuario_con_plataforma": "Usuario de otra plataforma",
            "perfil_social": "Enlace a un perfil en otra plataforma",
        },
    ),
    (
        CREDENCIALES,
        "credencial",
        {
            "aws_access_key": "Posible clave de acceso de AWS publicada",
            "google_api_key": "Posible clave de API de Google publicada",
            "github_token": "Posible token de GitHub publicado",
            "slack_token": "Posible token de Slack publicado",
            "stripe_key": "Posible clave secreta de Stripe publicada",
            "api_key_llm": "Posible clave de API de un servicio de IA publicada",
            "telegram_bot": "Posible token de bot de Telegram publicado",
            "jwt": "Posible token JWT publicado",
            "clave_privada": "Posible clave privada publicada",
            "url_con_credenciales": "URL con usuario y contraseña incrustados",
            "contrasena_declarada": "Contraseña escrita en el texto",
        },
    ),
)

_COMPILADOS = tuple(
    (tuple((etq, re.compile(patron), val) for etq, patron, val in entradas), categoria, descripciones)
    for entradas, categoria, descripciones in BANCOS
)

# Los valores de estas categorías NUNCA se reproducen: solo su tipo.
_SIN_VALOR = frozenset({"credencial"})
# Los enlaces y usuarios públicos se muestran tal cual (ya son públicos).
_SIN_ENMASCARAR = frozenset({"handle", "usuario_con_plataforma", "perfil_social"})


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


def enmascarar(etiqueta: str, valor: str) -> str:
    if etiqueta == "email":
        local, _, dominio = valor.partition("@")
        return f"{local[:1]}***@{dominio}"
    if etiqueta == "codigo_postal":
        return valor[:2] + "***"
    if etiqueta == "coordenadas":
        lat, lon = (float(x) for x in re.split(r"\s*,\s*", valor.strip()))
        return f"{lat:.2f}, {lon:.2f}"
    if etiqueta == "direccion":
        return re.sub(r"\d", "*", valor)
    if etiqueta in ("iban", "ccc", "tarjeta", "imei", "mac", "ipv6"):
        return _enmascarar_alnum(valor, 4)
    return _enmascarar_alnum(valor, 3 if etiqueta in ("dni", "nie", "cif", "pasaporte", "seguridad_social", "matricula", "matricula_antigua", "referencia_catastral") else 2)


def detectar_identificadores(texto: str) -> list[Hallazgo]:
    """Identificadores directos de un texto, ya enmascarados. Sin
    duplicados (misma categoría y mismo valor enmascarado)."""
    hallazgos: list[Hallazgo] = []
    for entradas, categoria, descripciones in _COMPILADOS:
        for etiqueta, rx, validador in entradas:
            for m in rx.finditer(texto):
                grupo = 1 if rx.groups and m.group(1) is not None else 0
                dato = m.group(grupo)
                if validador is not None and not validador(dato):
                    continue
                confianza = CONF_VALIDADO if validador is not None else CONF_FORMATO
                if categoria == "cuenta_externa":
                    confianza = CONF_ENLACE
                descripcion = descripciones[etiqueta]
                if categoria in _SIN_VALOR:
                    valor = descripcion
                elif etiqueta in _SIN_ENMASCARAR:
                    valor = f"{descripcion}: {dato}"
                else:
                    valor = f"{descripcion}: {enmascarar(etiqueta, dato)}"
                hallazgos.append(Hallazgo(categoria, valor, confianza))
    return list(dict.fromkeys(hallazgos))
