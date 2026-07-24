"""
Deriva dinámicamente el origen (esquema + host) de la petición entrante.

Contexto: antes, tanto el `redirect_uri` de Instagram como la URL a la que
se redirige al usuario tras un login correcto (`FRONTEND_ORIGIN`) se leían
de variables de entorno fijas. Eso obligaba a editar `.env` cada vez que
`cloudflared` (túnel rápido de Cloudflare, gratuito) generaba una URL
nueva -- cosa que pasa en cada reinicio del túnel.

Con webapp y backend sirviéndose bajo el MISMO origen (ver
webapp/nginx.conf, que hace de proxy hacia el backend), ese origen es
exactamente el Host que ve el backend en cada petición -- así que ya no
hace falta configurarlo a mano: se lee directamente de la petición.

OJO -- esto SOLO es correcto cuando frontend y backend comparten origen de
verdad (el montaje con Docker + nginx + túnel, o un despliegue detrás de un
único dominio). Si los sirves en puertos distintos sin proxy delante (p.
ej. `npm run dev` en 5173 + `uvicorn` suelto en 3000, sin Docker), esta
derivación NO sirve para `FRONTEND_ORIGIN` -- ahí la petición que llega al
backend trae el Host del propio backend (3000), no el de la SPA (5173).
Para ese caso, configura `FRONTEND_ORIGIN` explícitamente en `.env`, que
sigue teniendo prioridad (ver `frontend_target`).
"""
from fastapi import HTTPException, Request


def current_origin(request: Request) -> str:
    """Esquema+host de la petición actual, asumiendo siempre HTTPS (todo
    este proyecto exige HTTPS de cara al navegador -- cookies de sesión
    `https_only=True`, y Meta no acepta redirect_uri en http:// salvo
    casos especiales que no usamos aquí)."""
    host = request.headers.get("host")
    if not host:
        raise HTTPException(
            status_code=503,
            detail="No se pudo determinar el origen de la petición (sin cabecera Host).",
        )
    return f"https://{host}"


def frontend_target(request: Request, configured: str | None) -> str:
    """URL base a la que redirigir tras un login (correcto o denegado).

    Prioridad: `configured` (normalmente `settings.frontend_origin`) si
    viene fijado explícitamente -- pensado para producción con dominio
    propio, o desarrollo local SIN proxy compartido (frontend y backend en
    puertos distintos). Si no, se deriva del Host de la propia petición.
    """
    return configured or current_origin(request)
