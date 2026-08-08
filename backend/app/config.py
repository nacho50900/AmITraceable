"""
Configuración centralizada de la aplicación.

Importante (cumplimiento RGPD / diseño del TFG):
- No hay base de datos. Todo el estado vive en la sesión firmada del navegador
  (cookie) o en memoria durante la duración de la petición.
- Las credenciales de Reddit se leen de variables de entorno, nunca se
  hardcodean ni se loguean.
"""
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_photo_analysis_concurrency() -> int:
    """Punto de partida razonable para `photo_analysis_concurrency` en la
    máquina donde arranca el proceso, sin necesidad de tocar `.env` al
    cambiar de servidor.

    No es un óptimo calculado (eso solo se puede medir empíricamente, ver
    `app/performance_log.py` y `scripts/analyze_performance_log.py`): es
    una heurística de "reparto a partes iguales" -- 2 hilos de PyTorch por
    foto en vuelo es el punto donde, en la práctica, una sola inferencia ya
    no se beneficia mucho de más hilos (rendimientos decrecientes), así
    que por debajo de eso es mejor meter más fotos en paralelo que más
    hilos por foto. Con 4 núcleos da concurrencia 2 (como el valor fijo
    anterior); con 16 da 8. Nunca menos de 1.
    """
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count // 2)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    reddit_client_id: str
    reddit_client_secret: str
    reddit_redirect_uri: str
    reddit_user_agent: str = "tfg-identity-exposure-tool/0.1"

    # Instagram es opcional: si no se rellenan estas variables, el módulo de
    # Instagram simplemente no funcionará, pero el resto de la app (Reddit)
    # sigue operativa sin necesidad de tener estas credenciales.
    instagram_app_id: str | None = None
    instagram_app_secret: str | None = None
    instagram_redirect_uri: str | None = None

    session_secret_key: str
    # Opcional. Si no se fija, se deriva dinámicamente del Host de cada
    # petición para la redirección final tras el login -- ver
    # app/auth/dynamic_origin.py. Para CORS (que sí necesita un valor fijo
    # en el arranque, no puede ser dinámico por petición) se sigue usando
    # "http://localhost:5173" como valor por defecto si esto queda vacío.
    frontend_origin: str | None = None

    # Análisis con IA (opcional): si no se rellena, el botón "Analizar con
    # IA" del frontend simplemente devuelve "no disponible" en vez de
    # fallar. Se usa el tier GRATUITO de Mistral AI (La Plateforme) --
    # proveedor europeo (Francia), evitando transferencias internacionales
    # de datos personales fuera de la UE (RGPD Cap. V). El tier gratuito
    # tiene límite de peticiones/minuto y un tope mensual de tokens; al
    # agotarse, la API devuelve 429 y este módulo lo trata como
    # "no disponible ahora mismo", sin reintentar (para no arriesgar coste
    # ni spamear la cuota).
    mistral_api_key: str | None = None
    mistral_model: str = "mistral-small-latest"

    # Límites de extracción para no machacar las APIs y acotar el volumen de
    # datos procesados (principio de minimización de datos, RGPD).
    max_posts: int = 200
    max_comments: int = 300
    max_media: int = 200

    # Nº de fotos que se analizan EN PARALELO con los modelos de visión
    # (DINOv2 + Moondream2, ver app/vision/geolocation.py). DINOv2 es
    # rápido (solo un embedding) pero Moondream2 es lento (generación
    # autoregresiva) -- con concurrencia 1, mientras Moondream2 trabaja en
    # una foto, el núcleo que habría usado DINOv2 para la siguiente queda
    # ocioso. Subir esto aprovecha esos huecos, a costa de más RAM (cada
    # foto en vuelo mantiene su propia pasada de Moondream2 en memoria) y
    # de competir más por los mismos núcleos. Se calcula solo a partir de
    # los núcleos disponibles en la máquina donde arranca el proceso (ver
    # `_default_photo_analysis_concurrency`) -- no hace falta tocar esto al
    # cambiar de servidor. Sigue siendo sobreescribible con la variable de
    # entorno PHOTO_ANALYSIS_CONCURRENCY si algún día conviene forzar un
    # valor concreto (p. ej. para comparar configuraciones, ver
    # scripts/analyze_performance_log.py).
    photo_analysis_concurrency: int = Field(default_factory=_default_photo_analysis_concurrency)

    @field_validator("photo_analysis_concurrency", mode="before")
    @classmethod
    def _empty_env_value_means_auto(cls, value: object) -> object:
        """`PHOTO_ANALYSIS_CONCURRENCY=` (vacía) en `.env` llega aquí como
        cadena vacía, no como "variable no definida" -- sin este paso,
        Pydantic intentaría convertir "" a int y el arranque fallaría con
        un error de validación. Se trata igual que dejarla sin poner: se
        recalcula con `_default_photo_analysis_concurrency()`."""
        if value in (None, ""):
            return _default_photo_analysis_concurrency()
        return value

    # Interruptor para el análisis de CONTENIDO visual con Moondream2 (ver
    # app/vision/scene_analysis.py y `_maybe_analyze_content` en
    # geolocation.py). Desactivado por defecto: en máquinas con poca RAM
    # (menos de ~10-12GB libres) el modelo en float32 no cabe en memoria
    # junto al resto de servicios -- ver ADR correspondiente en docs/.
    # La geolocalización (DINOv2) no depende de esto y sigue funcionando
    # igual, activado o no.
    enable_scene_analysis: bool = False

    # Interruptor GENERAL de los logs de rendimiento (ver
    # app/performance_log.py y app/analysis_run_log.py): activado por
    # defecto porque son datos puramente técnicos, sin nada personal (ver
    # el docstring de cada módulo), y son la fuente empírica para la
    # sección "Plan de evaluación pendiente" de la memoria del TFG -- no
    # hay motivo para tenerlos desactivados salvo que se quiera evitar por
    # completo la escritura a disco (p. ej. un despliegue de solo lectura,
    # o simplemente no querer acumular estos ficheros). Con esto en false,
    # ninguno de los dos logs escribe nada, sin que el análisis en sí se
    # vea afectado.
    enable_performance_logging: bool = True


settings = Settings()
