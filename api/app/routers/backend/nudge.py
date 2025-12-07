from fastapi import APIRouter, HTTPException, Body, Depends
from ...services.backend_services.nudge_service import get_nudge_service
from ...schemas.backend.nudge import (
    FCMTokenRequest, 
    FCMTokenResponse, 
    SendCheckinRequest, 
    SendCheckinResponse,
    NotificationType
)
from ...exceptions import (
    UserNotFoundError,
    TokenNotFoundError,
    NotificationDisabledError,
    NotificationError
)
from app.core.security import get_current_user, CurrentUser

nudge_router = APIRouter()
nudge_service = get_nudge_service()  # Use singleton to prevent duplicate schedulers

@nudge_router.post("/register-fcm-token", response_model=FCMTokenResponse)
async def register_fcm_token(
    token_data: FCMTokenRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Register or update a user's FCM token for push notifications.
    
    Note: User must exist in the database before registering FCM token.
    """
    # Basic validation - service method will do detailed validation
    trimmed_token = token_data.fcm_token.strip() if token_data.fcm_token else ""
    if not trimmed_token:
        return FCMTokenResponse(
            success=False, 
            message="Invalid FCM token format. Token cannot be empty."
        )
    
    result = await nudge_service.save_fcm_token(current_user.email, trimmed_token)
    if result:
        return FCMTokenResponse(
            success=True, 
            message="FCM token registered successfully."
        )
    else:
        # Service method returns False for: user not found OR invalid token format
        # Check which one by looking at user existence
        from ...services.backend_services.db import get_db
        db = get_db()
        user_exists = await db["users"].find_one({"email": current_user.email})
        if not user_exists:
            return FCMTokenResponse(
                success=False, 
                message="Failed to register FCM token. User not found. Please ensure the user is registered first."
            )
        else:
            return FCMTokenResponse(
                success=False, 
                message="Failed to register FCM token. Token format is invalid. FCM tokens must be 100+ characters and contain only valid characters."
            )

@nudge_router.post("/send-checkin", response_model=SendCheckinResponse)
async def send_checkin_notification(
    request: SendCheckinRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Manually trigger one of the 4 check-in notifications (morning, checkin, evening, night).
    This endpoint allows you to immediately send a notification to a user by selecting the type.
    """
    try:
        if request.notification_type == NotificationType.MORNING:
            await nudge_service.send_morning_notification(current_user.email)
            return SendCheckinResponse(
                success=True,
                message="Morning notification sent successfully."
            )
        elif request.notification_type == NotificationType.CHECKIN:
            await nudge_service.send_checkin_notification(current_user.email)
            return SendCheckinResponse(
                success=True,
                message="Check-in notification sent successfully."
            )
        elif request.notification_type == NotificationType.EVENING:
            await nudge_service.send_evening_notification(current_user.email)
            return SendCheckinResponse(
                success=True,
                message="Evening notification sent successfully."
            )
        elif request.notification_type == NotificationType.NIGHT:
            await nudge_service.send_night_notification(current_user.email)
            return SendCheckinResponse(
                success=True,
                message="Night notification sent successfully."
            )
        else:
            return SendCheckinResponse(
                success=False,
                message=f"Invalid notification type: {request.notification_type}"
            )
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except TokenNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotificationDisabledError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except NotificationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send notification: {str(e)}")
