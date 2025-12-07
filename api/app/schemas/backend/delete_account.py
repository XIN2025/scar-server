from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime
from enum import Enum


class DeleteAccountStatus(str, Enum):
    """Status of account deletion request"""
    PENDING = "pending"  # Awaiting email confirmation
    CONFIRMED = "confirmed"  # Email confirmed, in grace period
    DELETED = "deleted"  # Permanently deleted after grace period
    CANCELLED = "cancelled"  # User cancelled deletion request


class DeleteAccountRequest(BaseModel):
    """Request to initiate account deletion"""
    user_email: Optional[EmailStr] = None  # Set from token, not from request
    reason: Optional[str] = Field(None, max_length=500, description="Optional reason for deletion")


class DeleteAccountConfirmationRequest(BaseModel):
    """Request to confirm account deletion via email token"""
    user_email: Optional[EmailStr] = None  # Set from token, not from request
    confirmation_token: str = Field(..., min_length=1, description="Confirmation token from email")


class DeleteAccountCancelRequest(BaseModel):
    """Request to cancel pending account deletion"""
    user_email: Optional[EmailStr] = None  # Set from token, not from request
    cancellation_token: str = Field(..., min_length=1, description="Cancellation token from email")


class DeleteAccountResponse(BaseModel):
    """Response for account deletion operations"""
    success: bool
    message: str
    deletion_scheduled_at: Optional[datetime] = None
    permanent_deletion_at: Optional[datetime] = None
    grace_period_days: int = 7

