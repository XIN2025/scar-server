from pydantic import BaseModel, EmailStr, field_validator
from typing import List
from zoneinfo import ZoneInfo

class UserRegister(BaseModel):
    name: str
    email: EmailStr
    timezone: str
    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, tz: str) -> str:
        normalized = tz.replace(" ", "_")
        try:
            ZoneInfo(normalized)
        except Exception:
            raise ValueError("Invalid IANA timezone")
        return normalized
class UserLogin(BaseModel):
    email: EmailStr

class OTPVerify(BaseModel):
    email: EmailStr
    otp: str

class UserPreferences(BaseModel):
    email: EmailStr
    age: int
    gender: str
    healthGoals: List[str]
    conditions: List[str]
    atRiskConditions: List[str]
    communicationStyle: str
    notifications: bool 