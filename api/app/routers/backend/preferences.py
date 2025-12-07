from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
from ...schemas.backend.preferences import (
    PillarType,
    TimePreference,
    PillarTimePreferences,
    TimePreferenceResponse
)
from ...services.backend_services.preferences_service import PreferencesService
from pydantic import EmailStr
from app.core.security import get_current_user, CurrentUser

preferences_router = APIRouter()
preferences_service = PreferencesService()

@preferences_router.post("/api/preferences/time", response_model=TimePreferenceResponse)
async def set_time_preferences(
    preferences: PillarTimePreferences,
    current_user: CurrentUser = Depends(get_current_user)
):
    try:
        # Set user_email from token
        preferences.user_email = current_user.email
        success = await preferences_service.set_time_preferences(
            current_user.email,
            preferences.preferences
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to set time preferences")
        return TimePreferenceResponse(
            success=True,
            message="Time preferences set successfully",
            data={"preferences": preferences.dict()}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@preferences_router.get("/api/preferences/time", response_model=TimePreferenceResponse)
async def get_time_preferences(current_user: CurrentUser = Depends(get_current_user)):
    try:
        preferences = await preferences_service.get_time_preferences(current_user.email)
        if not preferences:
            return TimePreferenceResponse(
                success=True,
                message="No time preferences found",
                data={"preferences": None}
            )
        return TimePreferenceResponse(
            success=True,
            message="Time preferences retrieved successfully",
            data={"preferences": preferences.dict()}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@preferences_router.put("/api/preferences/time/{pillar}", response_model=TimePreferenceResponse)
async def update_pillar_preference(
    pillar: PillarType,
    preference: TimePreference,
    current_user: CurrentUser = Depends(get_current_user)
):
    try:
        success = await preferences_service.update_pillar_preference(current_user.email, pillar, preference)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update pillar preference")
        return TimePreferenceResponse(
            success=True,
            message=f"Time preference for {pillar} updated successfully",
            data={"pillar": pillar, "preference": preference.dict()}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@preferences_router.delete("/api/preferences/time/{pillar}", response_model=TimePreferenceResponse)
async def delete_pillar_preference(
    pillar: PillarType,
    current_user: CurrentUser = Depends(get_current_user)
):
    try:
        success = await preferences_service.delete_pillar_preference(current_user.email, pillar)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete pillar preference")
        return TimePreferenceResponse(
            success=True,
            message=f"Time preference for {pillar} deleted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
