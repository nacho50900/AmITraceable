"""
Configuración centralizada de la aplicación.

Importante (cumplimiento RGPD / diseño del TFG):
- No hay base de datos. Todo el estado vive en la sesión firmada del navegador
  (cookie) o en memoria durante la duración de la petición.
- Las credenciales de Reddit se leen de variables de entorno, nunca se
  hardcodean ni se loguean.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    frontend_origin: str = "http://localhost:5173"

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


settings = Settings()
