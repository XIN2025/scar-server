from datetime import datetime
from typing import Dict, Optional
from ...schemas.backend.preferences import PillarType, TimePreference, PillarTimePreferences
from .db import get_db
import logging

logger = logging.getLogger(__name__)

class PreferencesService:
    def __init__(self):
        self.db = get_db()
        self.preferences_collection = self.db["pillar_preferences"]

    async def set_time_preferences(self, user_email: str, preferences: Dict[PillarType, TimePreference]) -> bool:
        """Set time preferences for multiple pillars."""
        try:
            # Convert all time objects to strings for storage
            preferences_dict = {}
            for pillar, pref in preferences.items():
                # preferred_time is already a validated "HH:MM" string in the schema
                preferences_dict[pillar.value] = {
                    **pref.dict(),
                    "preferred_time": pref.preferred_time
                }
            
            # Fetch existing preferences and merge with new ones to avoid overwriting
            existing = await self.preferences_collection.find_one({"user_email": user_email})
            existing_prefs = existing.get("preferences", {}) if existing else {}
            existing_prefs.update(preferences_dict)
            result = await self.preferences_collection.update_one(
                {"user_email": user_email},
                {"$set": {"preferences": existing_prefs, "updated_at": datetime.utcnow()}},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error setting time preferences: {str(e)}")
            return False

    async def get_time_preferences(self, user_email: str) -> Optional[PillarTimePreferences]:
        """Get time preferences for all pillars for a user."""
        try:
            preferences = await self.preferences_collection.find_one({"user_email": user_email})
            if not preferences:
                return None

            # Convert stored time strings back to time objects
            converted_preferences = {}
            for pillar, pref in preferences["preferences"].items():
                # Keep preferred_time as validated string per schema
                pref_copy = pref.copy()
                converted_preferences[PillarType(pillar)] = TimePreference(**pref_copy)

            return PillarTimePreferences(
                user_email=user_email,
                preferences=converted_preferences
            )
        except Exception as e:
            logger.error(f"Error getting time preferences: {str(e)}")
            return None

    async def update_pillar_preference(self, user_email: str, pillar: PillarType, preference: TimePreference) -> bool:
        """Update time preference for a specific pillar."""
        try:
            # Convert time object to string for storage
            # preferred_time is already a validated "HH:MM" string in the schema
            preference_dict = {
                **preference.dict(),
                "preferred_time": preference.preferred_time
            }
            
            result = await self.preferences_collection.update_one(
                {"user_email": user_email},
                {
                    "$set": {
                        f"preferences.{pillar.value}": preference_dict,
                        "updated_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error updating pillar preference: {str(e)}")
            return False

    async def delete_pillar_preference(self, user_email: str, pillar: PillarType) -> bool:
        """Delete time preference for a specific pillar."""
        try:
            result = await self.preferences_collection.update_one(
                {"user_email": user_email},
                {
                    "$unset": {f"preferences.{pillar.value}": ""},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting pillar preference: {str(e)}")
            return False
