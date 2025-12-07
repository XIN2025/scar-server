from fastapi import APIRouter, HTTPException, Depends

from app.services.backend_services.health_alert_service import (
    get_health_alert_service,
)
from app.schemas.utils import HttpResponse
from app.schemas.backend.health_alert import HealthMetricData
from app.core.security import get_current_user, CurrentUser

health_alert_router = APIRouter()
health_alert_service = get_health_alert_service()


@health_alert_router.post(
    "/hourly-data", response_model=HttpResponse
)
async def upload_hourly_health_data(
    data: HealthMetricData,
    current_user: CurrentUser = Depends(get_current_user)
):
    try:
        health_data = await health_alert_service.store_hourly_health_data(current_user.email, data)
        return HttpResponse(success=True, message="Health data uploaded successfully", data={"health_data": health_data.model_dump()})
    except Exception as e:
        return HttpResponse(success=False, message=str(e), data=None)


@health_alert_router.get(
    "/active", response_model=HttpResponse
)
async def get_active_health_alerts(current_user: CurrentUser = Depends(get_current_user)):
    try:
        alerts = await health_alert_service.get_active_health_alerts(current_user.email)
        return HttpResponse(
            success=True,
            message="Active alerts fetched successfully",
            data={"alerts": [alert.model_dump() for alert in alerts]},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching active alerts")

@health_alert_router.post("/{health_alert_id}/resolve", response_model=HttpResponse)
async def resolve_health_alert(
    health_alert_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    try:
        # Verify ownership before resolving
        alerts = await health_alert_service.get_active_health_alerts(current_user.email)
        alert_ids = [str(alert.id) for alert in alerts]
        if health_alert_id not in alert_ids:
            raise HTTPException(status_code=403, detail="Not authorized to resolve this alert")
        
        success = await health_alert_service.mark_health_alert_resolve(health_alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Health alert not found or already resolved")
        return HttpResponse(success=True, message="Health alert resolved successfully")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to resolve health alert")