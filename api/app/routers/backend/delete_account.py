"""
Delete Account Router - API endpoints for account deletion.
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Query, Depends
from fastapi.responses import JSONResponse

from app.services.backend_services.delete_account_service import get_delete_account_service
from app.schemas.backend.delete_account import (
    DeleteAccountRequest,
    DeleteAccountConfirmationRequest,
    DeleteAccountCancelRequest,
    DeleteAccountResponse,
)
from app.schemas.utils import HttpResponse
from app.exceptions import (
    UserNotFoundError,
    AuthenticationError,
    ValidationException,
    RateLimitError,
    DatabaseError,
)
from app.core.security import get_current_user, CurrentUser

logger = logging.getLogger(__name__)

delete_account_router = APIRouter()


def get_client_ip(request: Request) -> Optional[str]:
    """Extract client IP address from request"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    if request.client:
        return request.client.host
    
    return None


@delete_account_router.delete("/api/delete-account", response_model=HttpResponse)
async def delete_account(
    request_data: DeleteAccountRequest,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Initiate account deletion process.
    
    Creates a deletion request and sends a confirmation email.
    The account will be permanently deleted after a 7-day grace period
    unless the user cancels the deletion.
    
    Returns:
        HttpResponse with deletion request details
    
    Raises:
        400: Invalid request or validation error
        401: User not authenticated or not verified
        404: User not found
        429: Rate limit exceeded
        500: Server error
    """
    user_email = current_user.email
    ip_address = get_client_ip(request)
    
    logger.info(
        f"[DELETE_ACCOUNT_API] Deletion initiation request from {user_email} (IP: {ip_address})"
    )
    
    try:
        # Set user_email in request_data from authenticated user
        request_data.user_email = user_email
        delete_service = get_delete_account_service()
        result = await delete_service.initiate_deletion(
            request=request_data,
            ip_address=ip_address
        )
        
        if result.get("success") is False:
            logger.info(f"[DELETE_ACCOUNT_API] Demo account deletion blocked for {user_email}")
            return HttpResponse(
                success=False,
                message=result["message"],
                data=None
            )
        
        logger.info(f"[DELETE_ACCOUNT_API] Deletion request initiated for {user_email}")
        
        return HttpResponse(
            success=True,
            message=result["message"],
            data={
                "deletion_scheduled_at": result.get("deletion_scheduled_at"),
                "permanent_deletion_at": result.get("permanent_deletion_at"),
                "grace_period_days": result.get("grace_period_days", 7),
                "status": result.get("status", "pending"),
                "confirmation_token": result.get("confirmation_token"),
                "cancellation_token": result.get("cancellation_token"),
            }
        )
    
    except UserNotFoundError as e:
        logger.warning(f"[DELETE_ACCOUNT_API] User not found: {user_email}")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "USER_NOT_FOUND",
                "message": str(e),
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except AuthenticationError as e:
        logger.warning(f"[DELETE_ACCOUNT_API] Authentication error for {user_email}: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "AUTHENTICATION_ERROR",
                "message": str(e),
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except ValidationException as e:
        logger.warning(f"[DELETE_ACCOUNT_API] Validation error for {user_email}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "VALIDATION_ERROR",
                "message": str(e),
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except RateLimitError as e:
        logger.warning(f"[DELETE_ACCOUNT_API] Rate limit exceeded for {user_email}: {str(e)}")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "RATE_LIMIT_EXCEEDED",
                "message": str(e),
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except DatabaseError as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Database error for {user_email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": "Failed to process deletion request. Please try again later.",
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except Exception as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Unexpected error for {user_email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
                "details": {"error_type": type(e).__name__}
            }
        )


@delete_account_router.get("/api/delete-account/confirm")
async def confirm_account_deletion_get(
    email: str = Query(..., description="User email from confirmation link"),
    token: str = Query(..., description="Confirmation token from email link"),
    request: Request = None,
):
    """
    GET endpoint for email confirmation link.
    
    This endpoint handles the confirmation link clicked from the email.
    Extracts email and token from query parameters and processes the confirmation.
    
    Returns:
        JSON response with confirmation status
    """
    logger.info(
        f"[DELETE_ACCOUNT_API] GET confirmation request for {email} (from email link)"
    )
    
    try:
        # Create request object from query parameters
        request_data = DeleteAccountConfirmationRequest(
            user_email=email,
            confirmation_token=token
        )
        
        delete_service = get_delete_account_service()
        result = await delete_service.confirm_deletion(request=request_data)
        
        logger.warning(f"[DELETE_ACCOUNT_API] Account deletion confirmed via GET for {email}")
        
        # Return JSON response (frontend can handle this or redirect)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": result["message"],
                "data": {
                    "permanent_deletion_at": result.get("permanent_deletion_at"),
                    "grace_period_days": result.get("grace_period_days", 7)
                }
            }
        )
    
    except ValidationException as e:
        logger.warning(f"[DELETE_ACCOUNT_API] Validation error confirming deletion for {email}: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "VALIDATION_ERROR",
                "message": str(e),
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except DatabaseError as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Database error confirming deletion for {email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "DATABASE_ERROR",
                "message": "Failed to confirm deletion. Please try again later.",
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except Exception as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Unexpected error confirming deletion for {email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
                "details": {"error_type": type(e).__name__}
            }
        )


@delete_account_router.post("/api/delete-account/confirm", response_model=HttpResponse)
async def confirm_account_deletion(
    request_data: DeleteAccountConfirmationRequest,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Confirm account deletion using email token.
    
    This endpoint is called when the user clicks the confirmation link in the email.
    After confirmation, the account enters a grace period before permanent deletion.
    
    Returns:
        HttpResponse with confirmation details
    
    Raises:
        400: Invalid or expired token
        500: Server error
    """
    # Set user_email from authenticated user
    request_data.user_email = current_user.email
    user_email = current_user.email
    ip_address = get_client_ip(request)
    
    logger.info(
        f"[DELETE_ACCOUNT_API] Deletion confirmation attempt for {user_email} (IP: {ip_address})"
    )
    
    try:
        delete_service = get_delete_account_service()
        result = await delete_service.confirm_deletion(request=request_data)
        
        logger.warning(f"[DELETE_ACCOUNT_API] Account deletion confirmed for {user_email}")
        
        return HttpResponse(
            success=True,
            message=result["message"],
            data={
                "permanent_deletion_at": result.get("permanent_deletion_at"),
                "grace_period_days": result.get("grace_period_days", 7)
            }
        )
    
    except ValidationException as e:
        logger.warning(f"[DELETE_ACCOUNT_API] Validation error confirming deletion for {user_email}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "VALIDATION_ERROR",
                "message": str(e),
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except DatabaseError as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Database error confirming deletion for {user_email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": "Failed to confirm deletion. Please try again later.",
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except Exception as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Unexpected error confirming deletion for {user_email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
                "details": {"error_type": type(e).__name__}
            }
        )


@delete_account_router.get("/api/delete-account/cancel")
async def cancel_account_deletion_get(
    email: str = Query(..., description="User email from cancellation link"),
    token: str = Query(..., description="Cancellation token from email link"),
    request: Request = None,
):
    """
    GET endpoint for email cancellation link.
    
    This endpoint handles the cancellation link clicked from the email.
    Extracts email and token from query parameters and processes the cancellation.
    
    Returns:
        JSON response with cancellation status
    """
    logger.info(
        f"[DELETE_ACCOUNT_API] GET cancellation request for {email} (from email link)"
    )
    
    try:
        # Create request object from query parameters
        request_data = DeleteAccountCancelRequest(
            user_email=email,
            cancellation_token=token
        )
        
        delete_service = get_delete_account_service()
        result = await delete_service.cancel_deletion(request=request_data)
        
        logger.info(f"[DELETE_ACCOUNT_API] Account deletion cancelled via GET for {email}")
        
        # Return JSON response (frontend can handle this or redirect)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": result["message"],
                "data": None
            }
        )
    
    except ValidationException as e:
        logger.warning(f"[DELETE_ACCOUNT_API] Validation error cancelling deletion for {email}: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "VALIDATION_ERROR",
                "message": str(e),
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except DatabaseError as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Database error cancelling deletion for {email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "DATABASE_ERROR",
                "message": "Failed to cancel deletion. Please try again later.",
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except Exception as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Unexpected error cancelling deletion for {email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
                "details": {"error_type": type(e).__name__}
            }
        )


@delete_account_router.post("/api/delete-account/cancel", response_model=HttpResponse)
async def cancel_account_deletion(
    request_data: DeleteAccountCancelRequest,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Cancel a pending account deletion request.
    
    This endpoint is called when the user clicks the cancellation link in the email.
    The account will be restored if it was in the grace period.
    
    Returns:
        HttpResponse with cancellation confirmation
    
    Raises:
        400: Invalid token or cannot cancel
        500: Server error
    """
    # Set user_email from authenticated user
    request_data.user_email = current_user.email
    user_email = current_user.email
    ip_address = get_client_ip(request)
    
    logger.info(
        f"[DELETE_ACCOUNT_API] Deletion cancellation attempt for {user_email} (IP: {ip_address})"
    )
    
    try:
        delete_service = get_delete_account_service()
        result = await delete_service.cancel_deletion(request=request_data)
        
        logger.info(f"[DELETE_ACCOUNT_API] Account deletion cancelled for {user_email}")
        
        return HttpResponse(
            success=True,
            message=result["message"],
            data=None
        )
    
    except ValidationException as e:
        logger.warning(f"[DELETE_ACCOUNT_API] Validation error cancelling deletion for {user_email}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "VALIDATION_ERROR",
                "message": str(e),
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except DatabaseError as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Database error cancelling deletion for {user_email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": "Failed to cancel deletion. Please try again later.",
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except Exception as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Unexpected error cancelling deletion for {user_email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
                "details": {"error_type": type(e).__name__}
            }
        )


@delete_account_router.get("/api/delete-account/status", response_model=HttpResponse)
async def get_deletion_status(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get the current status of account deletion request for a user.
    
    Returns the most recent deletion request status, including:
    - status: pending, confirmed, deleted, or cancelled
    - created_at: When the deletion request was created
    - scheduled_deletion_at: When deletion was scheduled (if pending)
    - permanent_deletion_at: When account will be permanently deleted (if confirmed)
    - grace_period_days: Number of days in grace period
    
    Returns:
        HttpResponse with deletion request status, or None if no request exists
    
    Raises:
        500: Server error
    """
    user_email = current_user.email
    logger.info(
        f"[DELETE_ACCOUNT_API] Status check request for {user_email}"
    )
    
    try:
        delete_service = get_delete_account_service()
        deletion_request = await delete_service.get_deletion_status(user_email)
        
        if deletion_request:
            return HttpResponse(
                success=True,
                message="Deletion request found",
                data={
                    "status": deletion_request.get("status"),
                    "created_at": deletion_request.get("created_at"),
                    "scheduled_deletion_at": deletion_request.get("scheduled_deletion_at"),
                    "permanent_deletion_at": deletion_request.get("permanent_deletion_at"),
                    "grace_period_days": delete_service.GRACE_PERIOD_DAYS,
                }
            )
        else:
            return HttpResponse(
                success=True,
                message="No deletion request found",
                data=None
            )
    
    except DatabaseError as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Database error getting status for {user_email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": "Failed to retrieve deletion status. Please try again later.",
                "details": e.details if hasattr(e, 'details') else {}
            }
        )
    
    except Exception as e:
        logger.error(
            f"[DELETE_ACCOUNT_API] Unexpected error getting status for {user_email}: {type(e).__name__}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
                "details": {"error_type": type(e).__name__}
            }
        )
