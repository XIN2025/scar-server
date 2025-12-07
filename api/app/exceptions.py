from typing import Optional, Dict, Any
from fastapi import Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "status_code": self.status_code
        }


class DatabaseError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=500,
            error_code="DATABASE_ERROR"
        )


class ValidationException(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=400,
            error_code="VALIDATION_ERROR"
        )


class NotificationError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=500,
            error_code="NOTIFICATION_ERROR"
        )


class NudgeError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=500,
            error_code="NUDGE_ERROR"
        )


class GoalNotFoundError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=404,
            error_code="GOAL_NOT_FOUND"
        )


class UserNotFoundError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=404,
            error_code="USER_NOT_FOUND"
        )


class TokenNotFoundError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=404,
            error_code="TOKEN_NOT_FOUND"
        )


class NotificationDisabledError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=403,
            error_code="NOTIFICATIONS_DISABLED"
        )


class OperationTimeoutError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=408,
            error_code="TIMEOUT_ERROR"
        )


class InvalidTimeFormatError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=400,
            error_code="INVALID_TIME_FORMAT"
        )


class ConfigurationError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=500,
            error_code="CONFIGURATION_ERROR"
        )


class AuthenticationError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=401,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=403,
            error_code="AUTHORIZATION_ERROR"
        )


class RateLimitError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED"
        )


class ServiceUnavailableError(AppException):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=503,
            error_code="SERVICE_UNAVAILABLE"
        )


def handle_unexpected_error(error: Exception, context: str = "") -> AppException:
    message = f"Unexpected error occurred"
    if context:
        message += f" in {context}"
    
    details = {
        "original_error": str(error),
        "error_type": type(error).__name__,
        "context": context
    }
    
    return AppException(
        message=message,
        details=details,
        status_code=500,
        error_code="UNEXPECTED_ERROR"
    )

async def custom_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


async def generic_exception_handler(request: Request, exc: Exception):
    wrapped_exception = handle_unexpected_error(exc, f"{request.method} {request.url}")
    return JSONResponse(
        status_code=wrapped_exception.status_code,
        content=wrapped_exception.to_dict()
    )
