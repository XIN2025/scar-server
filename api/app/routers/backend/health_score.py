from app.services.backend_services.health_score_service import get_health_score_service

from fastapi import APIRouter, HTTPException, Depends

from app.schemas.utils import HttpResponse
from app.core.security import get_current_user, CurrentUser

health_score_router = APIRouter()
health_score_service = get_health_score_service()

@health_score_router.get('/', response_model=HttpResponse)
async def get_health_score(current_user: CurrentUser = Depends(get_current_user)):
    try:
        score = await health_score_service.get_health_score(current_user.email)
        return HttpResponse(success=True, data={"health_score": score}, message="Health score fetched successfully")
    except Exception as e:
        # TODO: If user is not found and I keep it as Internal Server Error, the message doesn't get passed to the client
        # And if I do then there is a security concern, look into thi
        raise HTTPException(status_code=500, detail=str(e))