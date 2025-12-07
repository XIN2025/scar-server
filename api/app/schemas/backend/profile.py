from datetime import date
import re
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=100)
    phone_number: Optional[str] = Field(default=None)
    date_of_birth: Optional[str] = Field(default=None)  # Format: YYYY-MM-DD
    blood_type: Optional[str] = Field(default=None)
    notifications_enabled: Optional[bool] = Field(default=None)
    timezone: Optional[str] = Field(default=None)

    @field_validator('date_of_birth')
    @classmethod
    def validate_date_of_birth(cls, v):
        if v is None or v == "":
            return None
        try:
            year, month, day = map(int, v.split('-'))
            birth_date = date(year, month, day)
            today = date.today()
            if birth_date > today:
                raise ValueError("Date of birth cannot be in the future")
            if today.year - birth_date.year > 120:
                raise ValueError("Date of birth seems too old")
            return v
        except ValueError as e:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

    @field_validator('phone_number')
    @classmethod
    def validate_phone(cls, v):
        if v is None or v == "":
            return None
        pattern = re.compile(r'^\+?1?\d{10,14}$')
        if not pattern.match(v):
            raise ValueError("Invalid phone number format")
        return v

    @field_validator('blood_type')
    @classmethod
    def validate_blood_type(cls, v):
        if v is None or v == "":
            return None
        pattern = re.compile(r'^(A|B|AB|O)[+-]$')
        if not pattern.match(v):
            raise ValueError("Invalid blood type format")
        return v.upper()

    @field_validator('full_name')
    @classmethod
    def validate_name(cls, v):
        if v is None or v == "":
            return None
        v = v.strip()
        if not v:
            return None
        if not all(c.isalpha() or c.isspace() or c in "-'" for c in v):
            raise ValueError("Name can only contain letters, spaces, hyphens, and apostrophes")
        return v
