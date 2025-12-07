from fastapi import APIRouter, HTTPException, Query
from app.services.backend_services.app_version_service import get_app_version_service
from app.schemas.utils import HttpResponse

app_version_router = APIRouter()
app_version_service = get_app_version_service()

@app_version_router.get("/api/app-version", response_model=HttpResponse)
async def get_app_version(platform: str = Query(..., description="Platform: 'ios' or 'android'")):
    """
    Get the current app version for Force Update feature.
    
    Returns the latest version information including:
    - version: Current app version string
    - force_update_required: Whether this version requires force update
    - min_supported_version: Minimum supported version
    - release_notes: Release notes for this version
    """
    try:
        if platform.lower() not in ["ios", "android"]:
            raise HTTPException(status_code=400, detail="Platform must be 'ios' or 'android'")
        
        version = await app_version_service.get_current_version(platform.lower())
        
        if not version:
            return HttpResponse(
                success=False,
                message=f"No app version found for platform: {platform}",
                data=None
            )
        
        return HttpResponse(
            success=True,
            message="App version retrieved successfully",
            data={"version": version.model_dump()}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve app version: {str(e)}")

