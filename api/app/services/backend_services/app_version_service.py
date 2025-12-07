from app.services.backend_services.db import get_db
from app.schemas.backend.app_version import AppVersionCreate, AppVersion
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AppVersionService:
    def __init__(self):
        self.db = get_db()
        self.app_versions_collection = self.db["app_versions"]

    async def get_current_version(self, platform: str) -> Optional[AppVersion]:
        """Get the current app version for a platform (ios/android)"""
        try:
            version_doc = await self.app_versions_collection.find_one(
                {"platform": platform},
                sort=[("created_at", -1)]  # Get the most recent version
            )
            
            if not version_doc:
                logger.warning(f"No app version found for platform: {platform}")
                return None
            
            version_doc["id"] = str(version_doc["_id"])
            del version_doc["_id"]
            return AppVersion(**version_doc)
        except Exception as e:
            logger.error(f"Error fetching app version for platform {platform}: {str(e)}", exc_info=True)
            raise

    async def create_or_update_version(self, version_data: AppVersionCreate) -> AppVersion:
        """Create or update app version"""
        try:
            now = datetime.now(timezone.utc)
            
            # Check if version exists for this platform
            existing = await self.app_versions_collection.find_one({"platform": version_data.platform})
            
            version_dict = version_data.model_dump()
            
            if existing:
                # Update existing version
                version_dict["updated_at"] = now
                await self.app_versions_collection.update_one(
                    {"platform": version_data.platform},
                    {"$set": version_dict}
                )
                version_dict["id"] = str(existing["_id"])
                version_dict["created_at"] = existing.get("created_at", now)
                logger.info(f"Updated app version for platform {version_data.platform}: {version_data.version}")
            else:
                # Create new version
                version_dict["created_at"] = now
                version_dict["updated_at"] = now
                result = await self.app_versions_collection.insert_one(version_dict)
                version_dict["id"] = str(result.inserted_id)
                logger.info(f"Created app version for platform {version_data.platform}: {version_data.version}")
            
            return AppVersion(**version_dict)
        except Exception as e:
            logger.error(f"Error creating/updating app version: {str(e)}", exc_info=True)
            raise

# Global instance
_app_version_service = None

def get_app_version_service() -> AppVersionService:
    global _app_version_service
    if _app_version_service is None:
        _app_version_service = AppVersionService()
    return _app_version_service

