"""
Bancos temáticos de regex para señales de ESTILO DE VIDA en el texto del
propio usuario: aficiones, equipo del que es, mascotas, vehículo, vivienda,
rutina, hábitos, idiomas, ingresos y (con anclas estrictas) salud e ideología.

Complementa a `demographic_patterns.py`: aquello son campos INE que estrechan
población (k-anonimato); esto son señales que NO tienen tabla INE pero que el
informe debe mostrar porque hacen a la persona más reconocible ("toca el
piano, tiene un gato, es del Betis y va a turnos de noche"). Se añaden a la
lista de atributos inferidos DESPUÉS de calcular el score (ver
`text_signals.py`), así que no alteran `compute_score`.

Convenciones (las mismas que `demographic_patterns.py`):
- Patrones sobre texto normalizado (minúsculas, sin tildes) y por cláusula;
  las frases sobre terceros ("mi hermano es del Barça") se descartan con
  `dp.es_propio`.
- Un banco es una tupla de `(valor_legible, patron)`; cada acierto es un
  `Hallazgo` con su confianza base.
- Una simple mención del tema NO basta ("vi un concierto en la tele"): hace
  falta una frase-ancla de gusto/práctica en primera persona, igual que con
  `practica_deportiva`.
"""
import re
from dataclasses import dataclass

from app.nlp import demographic_patterns as dp

# Confianzas base (ver docstring del módulo: son señales blandas).
CONF_GUSTO = 0.5  # "me encanta X": interés declarado
CONF_PRACTICA = 0.7  # "toco la guitarra": práctica declarada
CONF_ETIQUETA = 0.35  # hashtag: señal débil
CONF_HECHO = 0.7  # "tengo un perro", "vivo de alquiler"
CONF_SENSIBLE = 0.6  # salud / ideología / sindicato (art. 9 RGPD)

NOTA_ART9 = " (dato de categoría especial, art. 9 RGPD)"


@dataclass(frozen=True)
class Hallazgo:
    categoria: str
    valor: str
    confianza: float


Banco = tuple[tuple[str, "re.Pattern[str]"], ...]


def _banco(*entradas: tuple[str, str]) -> Banco:
    return tuple((valor, re.compile(patron)) for valor, patron in entradas)


# ---------------------------------------------------------------------------
# Aficiones: ancla de gusto + vocabulario del tema
# ---------------------------------------------------------------------------

_ANCLA_GUSTO = re.compile(
    r"\b(?:me (?:encanta|encantan|apasiona|apasionan|gusta|gustan|flipa|flipan|fascina|fascinan|chifla|chiflan)|"
    r"soy (?:un |una )?(?:gran )?(?:fan|fanatic[oa]|aficionad[oa]|adict[oa]|amante|apasionad[oa]|forof[oa]) (?:de|a|del)|"
    r"mi (?:hobby|hobbie|afici[oa]n|pasion|vicio) (?:es|son)|mis (?:hobbies|aficiones|pasiones) son|"
    r"en mi tiempo libre|en mis ratos libres|cuando tengo tiempo libre|vivo para)\b"
)
_VENTANA_GUSTO = 110  # caracteres tras el ancla en los que se busca el tema

TEMAS_AFICION: Banco = _banco(
    ("Música", r"\b(?:musica|conciertos?|festivales?|cantar|karaoke|rock|rap|hip ?hop|reggaeton|jazz|flamenco|indie|metal|punk|techno|vinilos?)\b"),
    ("Cine y series", r"\b(?:cine|peliculas?|series|netflix|hbo|anime|documentales|cortometrajes)\b"),
    ("Lectura", r"\b(?:leer|lectura|libros?|novelas?|comics?|manga|poesia|literatura)\b"),
    ("Videojuegos", r"\b(?:videojuegos|gaming|gamer|play ?station|ps[45]|xbox|nintendo|steam|fortnite|minecraft|valorant|zelda|pokemon)\b"),
    ("Juegos de mesa y rol", r"\b(?:juegos de mesa|juegos de rol|rol|dungeons|warhammer|cartas)\b"),
    ("Viajar", r"\b(?:viajar|viajes|mochilear|escapadas|conocer mundo|conocer sitios nuevos|turismo)\b"),
    ("Cocina y gastronomía", r"\b(?:cocinar|cocina|reposteria|hornear|recetas|gastronomia|foodie|vino|cafe de especialidad|barbacoa)\b"),
    ("Fotografía", r"\b(?:fotografia|fotografiar|hacer fotos|sacar fotos|camaras?|analogica|revelar)\b"),
    ("Arte y manualidades", r"\b(?:dibujar|dibujo|pintar|pintura|ilustracion|acuarela|manualidades|ceramica|escultura|graffiti|museos|exposiciones|tatuajes)\b"),
    ("Escritura", r"\b(?:escribir|escritura|relatos|poemas)\b"),
    ("Baile", r"\b(?:bailar|baile|salsa|bachata|danza|ballet|sevillanas)\b"),
    ("Teatro", r"\b(?:teatro|musicales|monologos|impro)\b"),
    ("Naturaleza y aire libre", r"\b(?:naturaleza|montana|camping|acampar|huerto|jardineria|plantas|pescar|astronomia)\b"),
    ("Tecnología y programación", r"\b(?:tecnologia|programar|programacion|hacking|ciberseguridad|arduino|raspberry|robotica|gadgets|open source|linux|impresion 3d|drones?)\b"),
    ("Motor", r"\b(?:coches|motos|motor|formula 1|f1|motogp|tuning|rallies|mecanica|karts?)\b"),
    ("Moda y belleza", r"\b(?:moda|ropa|zapatillas|sneakers|maquillaje|skincare|belleza|peluqueria)\b"),
    ("Coleccionismo", r"\b(?:coleccionar|coleccionismo|figuras|cromos|sellos|monedas|funkos|lego)\b"),
    ("Voluntariado y activismo", r"\b(?:voluntariado|ong|activismo|causas sociales|medio ambiente|animalismo|proteccion animal)\b"),
    ("Astrología, tarot o meditación", r"\b(?:astrologia|tarot|esoterismo|meditacion|mindfulness)\b"),
)

PRACTICA_DIRECTA: Banco = _banco(
    ("Toca un instrumento", r"\b(?:toco|tocando|aprendiendo a tocar) (?:la |el |un |una )?(?:guitarra|piano|bateria|violin|bajo|saxofon|flauta|trompeta|ukelele|teclado|violonchelo|acordeon|gaita|tambor)\b"),
    ("Música", r"\b(?:canto en (?:un|el) coro|canto en un grupo|toco en un grupo|tengo un grupo de musica|soy dj|pincho)\b"),
    ("Conciertos y festivales", r"\b(?:voy a conciertos|voy a festivales|asisto a conciertos|acabo de volver del concierto)\b"),
    ("Videojuegos", r"\b(?:juego (?:a la |a las |a los |al |a )?(?:play|ps[45]|playstation|xbox|nintendo|switch|videojuegos|consola|fortnite|minecraft|lol|valorant|fifa|call of duty|wow|zelda|pokemon)|soy gamer|hago streams?|stream(?:eo|ing) en twitch)\b"),
    ("Juegos de mesa y rol", r"\b(?:juego (?:al |a )?(?:rol|dungeons|warhammer|juegos de mesa)|dirijo partidas de rol)\b"),
    ("Fotografía", r"\b(?:hago fotos|saco fotos|hago fotografia|soy fotograf[oa])\b"),
    ("Lectura", r"\b(?:leo (?:mucho|todos los dias|cada noche|cada dia|novelas|libros|comics|manga)|estoy leyendo|acabo de leer|mi libro favorito)\b"),
    ("Cocina y gastronomía", r"\b(?:cocino|hago reposteria|horneo|preparo recetas)\b"),
    ("Arte y manualidades", r"\b(?:dibujo|pinto|hago ilustraciones|hago ceramica|tejo|hago punto|bordo)\b"),
    ("Baile", r"\b(?:bailo|voy a clases de baile|hago baile)\b"),
    ("Escritura", r"\bescribo (?:relatos|poemas|novelas|un libro|un blog)\b"),
    ("Coleccionismo", r"\bcoleccion(?:o|amos)\b"),
    ("Teatro", r"\b(?:hago teatro|hago improvisacion|soy actor|soy actriz)\b"),
    ("Naturaleza y aire libre", r"\b(?:tengo un huerto|mi huerto|cuido mis plantas|mis plantas)\b"),
    ("Voluntariado y activismo", r"\b(?:soy voluntari[oa]|hago voluntariado|colaboro con una ong)\b"),
)

# Hashtags (Instagram): coincidencia exacta o, para palabras de 6+ letras,
# por subcadena ("foodielife" contiene "foodie"). Señal débil.
TEMAS_ETIQUETA: tuple[tuple[str, frozenset[str]], ...] = (
    ("Música", frozenset({"musica", "music", "concierto", "conciertos", "festival", "guitarra", "guitar", "rock", "indie"})),
    ("Videojuegos", frozenset({"gamer", "gaming", "videojuegos", "playstation", "xbox", "nintendo", "twitch", "pcgaming"})),
    ("Cocina y gastronomía", frozenset({"foodie", "receta", "recetas", "cocina", "gastronomia", "reposteria", "foodporn"})),
    ("Viajar", frozenset({"travel", "viajes", "viajar", "wanderlust", "mochilero", "traveler", "viajeros"})),
    ("Fotografía", frozenset({"fotografia", "photography", "fotografo", "streetphotography"})),
    ("Arte y manualidades", frozenset({"arte", "dibujo", "illustration", "ilustracion", "artwork", "acuarela"})),
    ("Lectura", frozenset({"libros", "lectura", "booktok", "bookstagram", "leer", "books"})),
    ("Baile", frozenset({"baile", "dance", "danza", "bailar"})),
    ("Motor", frozenset({"coches", "motos", "motogp", "formula1", "tuning", "carspotting"})),
    ("Naturaleza y aire libre", frozenset({"naturaleza", "nature", "montana", "camping", "huerto"})),
    ("Moda y belleza", frozenset({"moda", "fashion", "ootd", "maquillaje", "skincare"})),
)

# ---------------------------------------------------------------------------
# Equipo de fútbol: (nombre, provincia, "cortos" tras "soy del…", "únicos" en cualquier contexto)
# ---------------------------------------------------------------------------

_EQUIPOS: tuple[tuple[str, str, str, str], ...] = (
    ("Real Madrid", "Madrid", r"real madrid|madrid", r"madridista|real madrid|hala madrid"),
    ("FC Barcelona", "Barcelona", r"barca|barcelona|fc barcelona", r"culer|barcelonista|visca el barca|forca barca"),
    ("Atlético de Madrid", "Madrid", r"atletico|atleti|at madrid|atletico de madrid", r"colchonero|atleti"),
    ("Sevilla FC", "Sevilla", r"sevilla|sevilla fc", r"sevillista|sevilla fc"),
    ("Real Betis", "Sevilla", r"betis|real betis", r"betico|betica|verdiblanco|viva el betis"),
    ("Valencia CF", "Valencia", r"valencia|valencia cf", r"valencianista|valencia cf"),
    ("Athletic Club", "Bizkaia", r"athletic|athletic club|athletic de bilbao", r"athleticzale|aupa athletic"),
    ("Real Sociedad", "Gipuzkoa", r"real sociedad|la real", r"txuri urdin|realzale"),
    ("Celta de Vigo", "Pontevedra", r"celta|celta de vigo", r"celtista|celta de vigo"),
    ("Deportivo de La Coruña", "A Coruña", r"depor|deportivo|deportivo de la coruna", r"deportivista"),
    ("Real Oviedo", "Asturias", r"oviedo|real oviedo", r"oviedista|real oviedo|carbayon"),
    ("Sporting de Gijón", "Asturias", r"sporting|sporting de gijon", r"sportinguista|mareona|sporting de gijon"),
    ("Real Zaragoza", "Zaragoza", r"zaragoza|real zaragoza", r"zaragocista|real zaragoza"),
    ("Osasuna", "Navarra", r"osasuna", r"osasuna|rojillo"),
    ("Real Valladolid", "Valladolid", r"valladolid|real valladolid|pucela", r"real valladolid|pucelano"),
    ("Villarreal CF", "Castellón", r"villarreal", r"villarreal|groguet"),
    ("Espanyol", "Barcelona", r"espanyol|espanol", r"periquito|perico"),
    ("Rayo Vallecano", "Madrid", r"rayo|rayo vallecano", r"rayista|rayo vallecano"),
    ("Getafe CF", "Madrid", r"getafe", r"getafe cf"),
    ("Granada CF", "Granada", r"granada|granada cf", r"granada cf"),
    ("Málaga CF", "Málaga", r"malaga|malaga cf", r"malaguista|boqueron"),
    ("Cádiz CF", "Cádiz", r"cadiz|cadiz cf", r"cadista|cadiz cf"),
    ("UD Las Palmas", "Las Palmas", r"las palmas|ud las palmas", r"ud las palmas"),
    ("CD Tenerife", "Santa Cruz de Tenerife", r"tenerife|cd tenerife", r"cd tenerife|tinerfeno"),
    ("RCD Mallorca", "Baleares", r"mallorca|rcd mallorca", r"mallorquinista|rcd mallorca"),
    ("Deportivo Alavés", "Álava", r"alaves|deportivo alaves", r"babazorro|alaves"),
    ("Racing de Santander", "Cantabria", r"racing|racing de santander", r"racinguista|racing de santander"),
    ("Real Murcia", "Murcia", r"real murcia|murcia", r"real murcia"),
    ("Elche CF", "Alicante", r"elche|elche cf", r"elchista|elche cf"),
    ("Levante UD", "Valencia", r"levante|levante ud", r"levante ud"),
)
_ANCLA_EQUIPO = r"\b(?:soy|somos|hincha|fan|forofo|forofa|socio|socia|seguidor|seguidora|abonado|abonada)\s+(?:un |una )?(?:del|de la)\s+(?:club\s+)?"
EQUIPOS: tuple[tuple[str, str, "re.Pattern[str]", "re.Pattern[str]"], ...] = tuple(
    (
        nombre,
        provincia,
        re.compile(_ANCLA_EQUIPO + r"(?:" + cortos + r")\b"),
        re.compile(r"\b(?:" + unicos + r")\b"),
    )
    for nombre, provincia, cortos, unicos in _EQUIPOS
)

# ---------------------------------------------------------------------------
# Mascotas, vehículo, vivienda, rutina, hábitos, idiomas, ingresos
# ---------------------------------------------------------------------------

_ESPECIES = {
    "perro": "perro", "perra": "perro", "cachorro": "perro", "gato": "gato", "gata": "gato", "gatito": "gato",
    "conejo": "conejo", "hamster": "hámster", "pajaro": "pájaro", "loro": "loro", "tortuga": "tortuga",
    "caballo": "caballo", "pez": "pez", "cobaya": "cobaya", "huron": "hurón", "periquito": "periquito",
}
_MASCOTA_RE = re.compile(
    r"\b(?:tengo|adopte|adoptamos|tenemos|mi|nuestr[oa]|mis|nuestros)\s+(?:un |una |nuevo |nueva |dos |tres )?"
    r"(" + "|".join(_ESPECIES) + r")s?\b"
)
_NOMBRE_MASCOTA_RE = re.compile(r"\bse llama\s+([a-z]{2,15})\b")

VEHICULO: Banco = _banco(
    ("Tiene carnet de conducir", r"\b(?:me saque el carnet|tengo el carnet|tengo carnet|aprobe el (?:practico|carnet)|me he sacado el carnet)\b"),
    ("Tiene coche", r"\b(?:mi coche|tengo coche|tengo un coche|me compre un coche|mi nuevo coche)\b"),
    ("Tiene moto", r"\b(?:mi moto|tengo moto|tengo una moto|me compre una moto)\b"),
    ("Va en bici o patinete al día a día", r"\b(?:voy (?:al trabajo|a clase|a la uni) en (?:bici|patinete|bicicleta)|mi bici|mi patinete)\b"),
    ("Usa transporte público a diario", r"\b(?:voy (?:al trabajo|a clase|a la uni) en (?:metro|bus|autobus|tren|cercanias)|mi abono (?:de transporte|del metro|de bus))\b"),
)
_MARCAS = (
    "seat", "renault", "peugeot", "citroen", "toyota", "volkswagen", "audi", "bmw", "mercedes", "ford", "opel",
    "fiat", "nissan", "kia", "hyundai", "tesla", "honda", "yamaha", "vespa", "dacia", "skoda", "mini", "mazda",
)
_MARCA_RE = re.compile(
    r"\b(?:mi (?:(?:coche|moto|vehiculo)(?: es| era)? )?|tengo |conduzco |me compre )(?:un |una )?(?:coche |moto )?"
    r"(?:nuevo |nueva )?(?:de marca )?(" + "|".join(_MARCAS) + r")\b"
)

VIVIENDA: Banco = _banco(
    ("Vive de alquiler", r"\b(?:vivo de alquiler|pago (?:el |mi )?alquiler|mi alquiler|estoy de alquiler|alquile (?:un|una|mi) (?:piso|casa|habitacion)|busco piso de alquiler)\b"),
    ("Es propietario/a o tiene hipoteca", r"\b(?:compre (?:un |una |mi )?(?:piso|casa|vivienda)|mi hipoteca|pago (?:la |mi )?hipoteca|hipotecad[oa]|mi casa propia|firme la hipoteca|me hipoteque)\b"),
    ("Comparte piso", r"\b(?:comparto piso|vivo en un piso compartido|mis compis de piso|companer[oa]s? de piso)\b"),
    ("Vive con sus padres", r"\b(?:vivo con mis padres|vivo en casa de mis padres|sigo en casa de mis padres|vivo con mi madre|vivo con mi padre)\b"),
    ("Vive en residencia o colegio mayor", r"\bvivo en (?:una )?(?:residencia|colegio mayor)\b"),
    ("Vive en un pueblo", r"\bvivo en un (?:pueblo|pueblecito)\b"),
    ("Vive en casa con jardín o adosado", r"\b(?:mi jardin|mi terraza|vivo en un adosado|vivo en una casa con jardin|mi chalet)\b"),
)

RUTINA: Banco = _banco(
    ("Hace guardias", r"\b(?:tengo guardia|estoy de guardia|hago guardias|mis guardias)\b"),
    ("Teletrabaja", r"\b(?:teletrabajo|teletrabajando|trabajo en remoto|trabajo remoto|home ?office|hibrido)\b"),
    ("Trabaja en fin de semana o festivos", r"\b(?:trabajo (?:los )?(?:sabados|domingos|fines de semana|festivos)|curro (?:el )?(?:sabado|domingo|finde))\b"),
    ("Horario fijo semanal", r"\b(?:todos los|cada) (?:lunes|martes|miercoles|jueves|viernes|sabados?|domingos?)\s+(?:voy|tengo|entreno|salgo|hago|toca|quedo|juego)\b"),
    ("Entrena a primera hora o por la noche", r"\b(?:entreno|voy al gym|voy al gimnasio|salgo a correr)\s+(?:por las (?:mananas|noches)|a las [0-9]{1,2}|a primera hora|de madrugada)\b"),
    ("Duerme poco o trasnocha", r"\b(?:duermo poco|trasnocho|me acuesto tardisimo|noctambulo|noctambula)\b"),
    ("Estudia en horario de tarde o noche", r"\b(?:clases de tarde|clases por la tarde|voy a clase de noche|estudio por las noches)\b"),
)

HABITOS: Banco = _banco(
    ("Fuma o vapea", r"\b(?:fumo|soy fumador|soy fumadora|vapeo|mi vaper|un cigarro)\b"),
    ("Ha dejado de fumar", r"\b(?:deje de fumar|dejando de fumar|llevo \w+ (?:dias|meses|anos) sin fumar)\b"),
    ("Vegano/a o vegetariano/a", r"\b(?:soy vegan[oa]|soy vegetarian[oa]|me hice vegan[oa]|me hice vegetarian[oa]|sigo una dieta vegana|dieta vegetariana|soy flexitarian[oa])\b"),
    ("No bebe alcohol", r"\b(?:no bebo(?: alcohol)?|no tomo alcohol|soy abstemi[oa]|llevo \w+ (?:dias|meses|anos) sin beber|sobri[oa])\b"),
    ("Bebe alcohol con frecuencia", r"\b(?:salgo de fiesta todos|cada finde de fiesta|resaca)\b"),
    ("Madruga habitualmente", r"\b(?:soy madrugador|soy madrugadora|me levanto a las [3-6]|suena el despertador a las [3-6])\b"),
)

_LENGUAS = {
    "ingles": "inglés", "frances": "francés", "aleman": "alemán", "italiano": "italiano", "portugues": "portugués",
    "chino": "chino", "japones": "japonés", "arabe": "árabe", "ruso": "ruso", "catalan": "catalán",
    "euskera": "euskera", "gallego": "gallego", "coreano": "coreano", "rumano": "rumano",
}
_LENGUA_RE = re.compile(
    r"\b(?:hablo|domino|manejo|estoy aprendiendo|estudio|aprendiendo|nivel (?:[abc][12]) de|bilingue en|bilingue de)\s+"
    r"(?:un poco de |algo de |bastante |muy bien )?(" + "|".join(_LENGUAS) + r")\b"
)
_NIVEL_IDIOMA_RE = re.compile(r"\bnivel\s+([abc][12])\b")

_INGRESOS_RE = re.compile(
    r"\b(?:cobro|gano|mi sueldo es|mi salario es|sueldo de|salario de|me pagan)\s+(?:unos |aproximadamente |como |solo )?\d[\d.,]*\s*(?:€|euros?|k\b)"
)

# ---------------------------------------------------------------------------
# Categorías especiales (art. 9 RGPD): salud, ideología, sindicato
# ---------------------------------------------------------------------------

_CONDICIONES = (
    r"diabetes", r"cancer", r"depresion", r"ansiedad", r"bipolaridad|trastorno bipolar", r"esquizofrenia", r"tdah",
    r"autismo|tea\b", r"epilepsia", r"esclerosis(?: multiple)?", r"asma", r"alergia\w*", r"celiaquia|celiaco|celiaca",
    r"vih", r"fibromialgia", r"migranas?", r"insomnio", r"anorexia", r"bulimia", r"trastorno \w+", r"hipertension",
    r"colesterol alto", r"cronica", r"discapacidad",
)
CONDICIONES = _CONDICIONES  # reutilizado por context_patterns.py (salud de terceros)
_ANCLA_SALUD = (
    r"\b(?:tengo|padezco|sufro(?: de)?|sufro|me diagnosticaron|me han diagnosticado|me diagnostico|"
    r"estoy en tratamiento (?:de|por|contra)|mi diagnostico es)\s+(?:un |una |el |la |de )?(?:leve |grave |cronic[oa] )?"
)
SALUD: Banco = _banco(
    ("Menciona una condición de salud", _ANCLA_SALUD + r"(?:" + "|".join(_CONDICIONES) + r")\b"),
    ("Menciona atención de salud mental", r"\b(?:voy al psicologo|voy al psiquiatra|voy a terapia|mi psicologo|mi psicologa|mi psiquiatra|mi terapeuta|estoy en terapia)\b"),
    ("Menciona un embarazo", r"\b(?:estoy embarazada|estamos embarazados|estoy de \d+ (?:semanas|meses)|voy a ser (?:mama|papa|madre|padre))\b"),
    ("Menciona una baja médica o una operación", r"\b(?:estoy de baja(?: medica)?|estoy en baja|me operaron|me operan|me han operado|me van a operar|estoy ingresad[oa])\b"),
)

_PARTIDOS = (
    r"psoe", r"pp\b", r"vox", r"podemos", r"sumar", r"erc", r"junts", r"bildu", r"pnv", r"ciudadanos", r"cup\b",
    r"bng", r"compromis", r"mas madrid", r"coalicion canaria", r"izquierda unida|iu\b", r"upyd", r"pacma",
)
_ANCLA_VOTO = (
    r"\b(?:(?:voto|vote|votare|votaba|siempre voto|siempre he votado) (?:a|al|por)|soy (?:votante|militante|simpatizante|afiliad[oa]) (?:de|del|a)|"
    r"afiliad[oa] (?:a|al)|militante (?:de|del)|mi partido es(?: el)?)\s+(?:la |el )?"
)
_IDEOLOGIAS = (
    r"izquierdas?", r"derechas?", r"socialista", r"comunista", r"anarquista", r"liberal", r"conservador(?:a)?",
    r"republican[oa]", r"independentista", r"feminista", r"ecologista", r"progresista", r"nacionalista",
)
IDEOLOGIA: Banco = _banco(
    ("Declara su voto o afiliación política", _ANCLA_VOTO + r"(?:" + "|".join(_PARTIDOS) + r")"),
    ("Se declara de una ideología", r"\bsoy (?:un |una |muy |bastante )?(?:de )?(?:" + "|".join(_IDEOLOGIAS) + r")\b"),
    ("Declara afiliación sindical", r"\b(?:estoy afiliad[oa] a|afiliad[oa] a|soy delegad[oa] (?:de|del)|soy sindicalista)\s*(?:ccoo|ugt|cgt|uso|sindicato)?\b"),
)


# ---------------------------------------------------------------------------
# Detección
# ---------------------------------------------------------------------------


def _por_banco(banco: Banco, categoria: str, confianza: float, clausula: str) -> list[Hallazgo]:
    hallazgos: list[Hallazgo] = []
    for valor, patron in banco:
        match = patron.search(clausula)
        if match and dp.es_propio(clausula, match.start()):
            hallazgos.append(Hallazgo(categoria, valor, confianza))
    return hallazgos


def _aficiones(clausula: str) -> list[Hallazgo]:
    hallazgos = _por_banco(PRACTICA_DIRECTA, "aficion", CONF_PRACTICA, clausula)
    ancla = _ANCLA_GUSTO.search(clausula)
    if ancla and dp.es_propio(clausula, ancla.start()):
        cola = clausula[ancla.end() : ancla.end() + _VENTANA_GUSTO]
        hallazgos += [Hallazgo("aficion", tema, CONF_GUSTO) for tema, patron in TEMAS_AFICION if patron.search(cola)]
    return hallazgos


def _equipo(clausula: str) -> list[Hallazgo]:
    """Gana la coincidencia MÁS LARGA: 'soy socio del Real Oviedo' es el
    Oviedo, no el Madrid, aunque 'real' también sea alias corto de otro."""
    mejor: tuple[int, str, str] | None = None
    for nombre, provincia, con_ancla, unico in EQUIPOS:
        for patron in (con_ancla, unico):
            match = patron.search(clausula)
            if match and dp.es_propio(clausula, match.start()) and (mejor is None or len(match.group(0)) > mejor[0]):
                mejor = (len(match.group(0)), nombre, provincia)
    if mejor is None:
        return []
    _, nombre, provincia = mejor
    return [Hallazgo("aficion", f"Seguidor/a del {nombre} (equipo de {provincia})", CONF_GUSTO + 0.1)]


def _mascota(clausula: str) -> list[Hallazgo]:
    match = _MASCOTA_RE.search(clausula)
    if not match or not dp.es_propio(clausula, match.start()):
        return []
    especie = _ESPECIES[match.group(1)]
    nombre = _NOMBRE_MASCOTA_RE.search(clausula[match.end() :])
    valor = f"Tiene {especie}" + (f" (dice su nombre: {nombre.group(1).capitalize()})" if nombre else "")
    return [Hallazgo("mascota", valor, CONF_HECHO + (0.1 if nombre else 0.0))]


def _vehiculo(clausula: str) -> list[Hallazgo]:
    hallazgos = _por_banco(VEHICULO, "vehiculo", CONF_HECHO, clausula)
    marca = _MARCA_RE.search(clausula)
    if marca and dp.es_propio(clausula, marca.start()):
        hallazgos.append(Hallazgo("vehiculo", f"Menciona su vehículo de marca {marca.group(1).capitalize()}", CONF_HECHO))
    return hallazgos


def _idiomas(clausula: str) -> list[Hallazgo]:
    hallazgos: list[Hallazgo] = []
    for match in _LENGUA_RE.finditer(clausula):
        if not dp.es_propio(clausula, match.start()):
            continue
        nivel = _NIVEL_IDIOMA_RE.search(clausula)
        sufijo = f" (nivel {nivel.group(1).upper()})" if nivel else ""
        hallazgos.append(Hallazgo("idiomas", f"Habla o estudia {_LENGUAS[match.group(1)]}{sufijo}", CONF_HECHO - 0.05))
    return hallazgos


def _ingresos(clausula: str) -> list[Hallazgo]:
    match = _INGRESOS_RE.search(clausula)
    if match and dp.es_propio(clausula, match.start()):
        return [Hallazgo("ingresos", "Menciona su sueldo o ingresos", CONF_HECHO)]
    return []


def _sensibles(clausula: str) -> list[Hallazgo]:
    hallazgos = _por_banco(SALUD, "salud", CONF_SENSIBLE, clausula)
    hallazgos += _por_banco(IDEOLOGIA, "ideologia_politica", CONF_SENSIBLE, clausula)
    return [Hallazgo(h.categoria, h.valor + NOTA_ART9, h.confianza) for h in hallazgos]


def _etiquetas(etiquetas: list[str]) -> list[Hallazgo]:
    normalizadas = {dp.normalizar(e).lstrip("#") for e in etiquetas}
    hallazgos: list[Hallazgo] = []
    for tema, claves in TEMAS_ETIQUETA:
        if any(e in claves or any(len(k) >= 6 and k in e for k in claves) for e in normalizadas):
            hallazgos.append(Hallazgo("aficion", tema, CONF_ETIQUETA))
    return hallazgos


def detectar_estilo_de_vida(texto: str, etiquetas: list[str] | None = None) -> list[Hallazgo]:
    """Todas las señales de estilo de vida de un texto (y, opcionalmente,
    de sus hashtags). Sin duplicados dentro del mismo texto."""
    hallazgos: list[Hallazgo] = _etiquetas(etiquetas or [])
    for clausula in dp.clausulas(dp.normalizar(texto), separar_por_y=False):
        hallazgos += _aficiones(clausula)
        hallazgos += _equipo(clausula)
        hallazgos += _mascota(clausula)
        hallazgos += _vehiculo(clausula)
        hallazgos += _por_banco(VIVIENDA, "vivienda", CONF_HECHO, clausula)
        hallazgos += _por_banco(RUTINA, "rutina", CONF_HECHO - 0.1, clausula)
        hallazgos += _por_banco(HABITOS, "habito", CONF_HECHO - 0.1, clausula)
        hallazgos += _idiomas(clausula)
        hallazgos += _ingresos(clausula)
        hallazgos += _sensibles(clausula)
    return list(dict.fromkeys(hallazgos))
