from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Dict, Optional
from datetime import time
from enum import Enum

class PillarType(str, Enum):
    HEALTH = "HEALTH"
    FITNESS = "FITNESS"
    NUTRITION = "NUTRITION"
    MENTAL = "MENTAL"
    PERSONAL = "PERSONAL"

class TimePreference(BaseModel):
    preferred_time: str  # HH:mm format
    duration_minutes: int = Field(..., ge=5, le=240)  # Between 5 minutes and 4 hours
    days_of_week: list[int] = Field(..., description="List of days (0=Monday, 6=Sunday)")
    reminder_before_minutes: Optional[int] = Field(None, ge=0, le=60)

    @field_validator('preferred_time')
    @classmethod
    def validate_time_format(cls, v):
        try:
            # Check if string matches HH:mm format
            if not v or len(v.split(':')) != 2:
                raise ValueError("Time must be in HH:mm format")
            
            hours, minutes = map(int, v.split(':'))
            if not (0 <= hours <= 23 and 0 <= minutes <= 59):
                raise ValueError("Invalid hours or minutes")
                
            return f"{hours:02d}:{minutes:02d}"
        except ValueError as e:
            raise ValueError(str(e))
        except Exception:
            raise ValueError("Time must be in HH:mm format")

    @property
    def time_obj(self) -> time:
        hour, minute = map(int, self.preferred_time.split(':'))
        return time(hour=hour, minute=minute)

class PillarTimePreferences(BaseModel):
    user_email: Optional[EmailStr] = None  # Set from token, not from request
    preferences: Dict[PillarType, TimePreference]

    @field_validator('preferences')
    @classmethod
    def validate_preferences(cls, v):
        if not v:
            raise ValueError("At least one pillar preference is required")
        return v

class TimePreferenceResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
