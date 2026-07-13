import secrets

from fastapi import Depends, Header, HTTPException, Request

from app.core.config import Settings, get_settings


def require_admin_token(
    request: Request,
    x_clade_admin_token: str | None = Header(
        default=None, alias="X-Clade-Admin-Token"
    ),
    settings: Settings = Depends(get_settings),
) -> None:
    if request.method in {"GET", "HEAD"}:
        return

    configured_token = settings.clade_admin_token
    if not configured_token:
        raise HTTPException(status_code=503, detail="管理员功能未启用")

    if not secrets.compare_digest(x_clade_admin_token or "", configured_token):
        raise HTTPException(status_code=403, detail="管理员令牌无效")
