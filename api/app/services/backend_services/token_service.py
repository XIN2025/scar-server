import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.services.backend_services.db import get_db

REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))


class RefreshTokenService:
    def __init__(self) -> None:
        self.db = get_db()
        self.collection = self.db["refresh_tokens"]

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    async def create_refresh_token(
        self, user_email: str, token_version: int
    ) -> tuple[str, datetime]:
        """Create and persist a refresh token for a user."""
        raw_token = secrets.token_urlsafe(64)
        hashed = self._hash_token(raw_token)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        await self.collection.insert_one(
            {
                "token_hash": hashed,
                "user_email": user_email,
                "token_version": token_version,
                "created_at": now,
                "expires_at": expires_at,
                "revoked": False,
                "revoked_at": None,
                "replaced_by": None,
            }
        )

        return raw_token, expires_at

    async def _get_valid_token_doc(self, token: str) -> dict:
        hashed = self._hash_token(token)
        token_doc = await self.collection.find_one({"token_hash": hashed})
        if not token_doc:
            raise ValueError("Refresh token is invalid.")

        if token_doc.get("revoked"):
            raise ValueError("Refresh token has been revoked.")

        expires_at = token_doc.get("expires_at")
        if not expires_at:
            raise ValueError("Refresh token has expired.")
        
        # Ensure expires_at is timezone-aware (MongoDB might return naive datetime)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        
        now = datetime.now(timezone.utc)
        if expires_at < now:
            await self.collection.update_one(
                {"_id": token_doc["_id"]},
                {"$set": {"revoked": True, "revoked_at": now}},
            )
            raise ValueError("Refresh token has expired.")

        return token_doc

    async def rotate_refresh_token(
        self, token: str
    ) -> tuple[dict, str, datetime]:
        """Validate an existing refresh token and rotate it."""
        token_doc = await self._get_valid_token_doc(token)
        new_token, new_expires_at = await self.create_refresh_token(
            token_doc["user_email"], token_doc["token_version"]
        )

        await self.collection.update_one(
            {"_id": token_doc["_id"]},
            {
                "$set": {
                    "revoked": True,
                    "revoked_at": datetime.now(timezone.utc),
                    "replaced_by": self._hash_token(new_token),
                }
            },
        )

        return token_doc, new_token, new_expires_at

    async def revoke_all_tokens_for_user(self, user_email: str) -> None:
        """Revoke every refresh token associated with a user."""
        await self.collection.update_many(
            {"user_email": user_email, "revoked": False},
            {
                "$set": {
                    "revoked": True,
                    "revoked_at": datetime.now(timezone.utc),
                }
            },
        )


_refresh_token_service: Optional[RefreshTokenService] = None


def get_refresh_token_service() -> RefreshTokenService:
    global _refresh_token_service
    if _refresh_token_service is None:
        _refresh_token_service = RefreshTokenService()
    return _refresh_token_service

