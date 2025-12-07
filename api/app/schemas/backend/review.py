from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from app.schemas.backend.encrypt import EncryptedField

class ReviewCreate(BaseModel):
    """Schema for creating a customer review/feedback"""
    user_email: Optional[EmailStr] = None  # Set from token, not from request
    rating: int = Field(..., ge=1, le=5, description="Rating out of 5 stars")
    feedback: Optional[str] = EncryptedField(None, max_length=5000, description="Customer feedback text")

class Review(ReviewCreate):
    """Schema for review response"""
    id: str
    created_at: datetime

