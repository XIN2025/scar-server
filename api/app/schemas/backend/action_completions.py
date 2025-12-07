from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from datetime import datetime, date, time

class ActionItemCompletion(BaseModel):
    """Track completion of daily action items"""
    user_email: EmailStr
    goal_id: str
    action_item_title: str
    completion_date: datetime
    completed: bool = True
    notes: Optional[str] = Field(None, max_length=500)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ActionItemCompletionCreate(BaseModel):
    """Create completion record for an action item"""
    goal_id: str
    action_item_title: str
    completion_date: datetime
    completed: bool = True
    notes: Optional[str] = Field(None, max_length=500)
    
    @field_validator('completion_date', mode='before')
    @classmethod
    def parse_completion_date(cls, v):
        if isinstance(v, str):
            # If it's a date string (YYYY-MM-DD), convert to datetime at start of day
            if len(v) == 10 and '-' in v:  # YYYY-MM-DD format
                return datetime.fromisoformat(v + 'T00:00:00')
            # If it's a full datetime string, parse it
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        elif isinstance(v, date) and not isinstance(v, datetime):
            # Convert date to datetime at start of day
            return datetime.combine(v, time.min)
        return v

class ActionItemCompletionCreateRequest(BaseModel):
    """Request body for creating completion record (goal_id comes from URL)"""
    action_item_title: str
    completion_date: datetime
    completed: bool = True
    notes: Optional[str] = Field(None, max_length=500)
    
    @field_validator('completion_date', mode='before')
    @classmethod
    def parse_completion_date(cls, v):
        if isinstance(v, str):
            # If it's a date string (YYYY-MM-DD), convert to datetime at start of day
            if len(v) == 10 and '-' in v:  # YYYY-MM-DD format
                return datetime.fromisoformat(v + 'T00:00:00')
            # If it's a full datetime string, parse it
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        elif isinstance(v, date) and not isinstance(v, datetime):
            # Convert date to datetime at start of day
            return datetime.combine(v, time.min)
        return v

class ActionItemCompletionUpdate(BaseModel):
    """Update completion record"""
    completed: bool
    notes: Optional[str] = Field(None, max_length=500)

class DailyCompletionStats(BaseModel):
    """Daily completion statistics for a goal"""
    date: datetime
    total_scheduled_items: int
    completed_items: int
    completion_percentage: float
    action_items: list[str] = Field(default_factory=list)  # List of completed action item titles for this day

class WeeklyCompletionStats(BaseModel):
    """Weekly completion statistics for a goal"""
    goal_id: str
    week_start: datetime
    week_end: datetime
    total_scheduled_days: int
    completed_days: int
    daily_stats: list[DailyCompletionStats]
    overall_completion_percentage: float
