from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class AppVersionCreate(BaseModel):
    """Schema for creating/updating app version"""
    version: str = Field(..., description="App version string (e.g., '1.0.0')")
    platform: str = Field(..., description="Platform: 'ios' or 'android'")
    force_update_required: bool = Field(default=False, description="Whether this version requires force update")
    min_supported_version: Optional[str] = Field(None, description="Minimum supported version")
    release_notes: Optional[str] = Field(None, max_length=2000, description="Release notes for this version")

class AppVersion(AppVersionCreate):
    """Schema for app version response"""
    id: str
    created_at: datetime
    updated_at: datetime

