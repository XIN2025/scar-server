from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from fastapi import Depends, HTTPException, status, Header
from pydantic import BaseModel

import jwt
from app.config import SECRET_KEY
from app.services.backend_services.db import get_db

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    __import__("os").getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15")
)


class CurrentUser(BaseModel):
    """Current authenticated user information"""
    email: str
    name: str
    token_version: int


def create_access_token(
    data: Dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        ) from exc


def get_bearer_token(authorization: Optional[str] = Header(None)) -> str:
    """Extract Bearer token from Authorization header."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not authorization:
        logger.warning("[AUTH] Authorization header missing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        logger.warning(f"[AUTH] Invalid authorization scheme: {scheme}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme. Expected 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def get_current_user(token: str = Depends(get_bearer_token)) -> CurrentUser:
    """
    Dependency to get current authenticated user from JWT token.
    Verifies token signature, expiration, and token version.
    """
    # Decode and validate token
    payload = decode_access_token(token)
    
    email = payload.get("sub")
    name = payload.get("name")
    token_version = payload.get("token_version", 0)
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    # Verify user exists and token version matches
    db = get_db()
    users = db["users"]
    db_user = await users.find_one({"email": email})
    
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    if not db_user.get("verified", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account not verified",
        )
    
    # Check token version (for logout/revocation)
    current_version = db_user.get("token_version", 0)
    if token_version != current_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked. Please login again.",
        )
    
    return CurrentUser(
        email=email,
        name=name or db_user.get("name", ""),
        token_version=token_version,
    )

