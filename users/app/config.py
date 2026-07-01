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

    session_secret_key: str
    frontend_origin: str = "http://localhost:5173"

    # Límites de extracción para no machacar la API de Reddit y acotar el
    # volumen de datos procesados (principio de minimización de datos, RGPD).
    max_posts: int = 200
    max_comments: int = 300


settings = Settings()
