"""
API endpoints for personalization profile management
"""
from fastapi import APIRouter, HTTPException, Query, Depends
import logging

from app.core.security import get_current_user, CurrentUser
from app.services.ai_services.personalization_profile_service import get_personalization_profile_service

logger = logging.getLogger(__name__)

personalization_router = APIRouter(prefix="/personalization", tags=["personalization"])

@personalization_router.get("/profile")
async def get_profile(current_user: CurrentUser = Depends(get_current_user)):
    """
    Get cached personalization profile for the authenticated user.
    Returns cached profile if it exists, or status indicating why profile doesn't exist.
    """
    try:
        service = get_personalization_profile_service()
        
        profile = await service.get_cached_profile(current_user.email)
        
        if profile:
            return {
                "success": True,
                "profile": profile,
                "message": "Profile retrieved successfully"
            }
        else:
            should_generate, reason = await service.should_generate_profile(current_user.email)
            return {
                "success": False,
                "profile": None,
                "message": f"No profile available: {reason}",
                "can_generate": should_generate
            }
    except Exception as e:
        logger.error(f"Error retrieving profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@personalization_router.post("/profile/generate")
async def generate_profile(
    force: bool = Query(False),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Generate personalization profile for the authenticated user.
    
    - **force**: If True, regenerate even if valid profile exists
    """
    try:
        service = get_personalization_profile_service()
        
        profile = await service.generate_profile(current_user.email, force=force)
        
        return {
            "success": True,
            "profile": profile,
            "message": "Profile generated successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@personalization_router.get("/profile/status")
async def get_profile_status(current_user: CurrentUser = Depends(get_current_user)):
    """
    Get status of personalization profile for the authenticated user.
    Returns whether user qualifies for profile and if profile exists.
    """
    try:
        service = get_personalization_profile_service()
        
        has_health_data = await service._user_has_health_data(current_user.email)
        has_lab_reports = await service._user_has_lab_reports(current_user.email)
        
        profile = await service.get_cached_profile(current_user.email)
        
        should_generate, reason = await service.should_generate_profile(current_user.email)
        
        return {
            "success": True,
            "status": {
                "has_health_data": has_health_data,
                "has_lab_reports": has_lab_reports,
                "profile_exists": profile is not None,
                "profile_generated_at": profile.get("generated_at") if profile else None,
                "should_generate": should_generate,
                "reason": reason
            }
        }
    except Exception as e:
        logger.error(f"Error checking profile status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


