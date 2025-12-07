from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta, time
from enum import Enum

class ActionPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ActionTimeSlot(BaseModel):
    start_time: str  # HH:MM format
    end_time: str    # HH:MM format
    duration: str    # HH:MM:SS format
    pillar: Optional[str] = None
    frequency: Optional[str] = None
    priority: Optional[str] = None
    health_notes: Optional[List[str]] = None

class DailySchedule(BaseModel):
    date: str  # ISO format
    time_slots: List[ActionTimeSlot]
    total_duration: str  # HH:MM:SS format

class TimeEstimate(BaseModel):
    min_duration: timedelta
    max_duration: timedelta
    recommended_frequency: str  # e.g., "daily", "3 times per week", etc.
    time_of_day: Optional[str] = None  # e.g., "morning", "evening", etc.

class WeeklyActionSchedule(BaseModel):
    monday: Optional[DailySchedule] = None
    tuesday: Optional[DailySchedule] = None
    wednesday: Optional[DailySchedule] = None
    thursday: Optional[DailySchedule] = None
    friday: Optional[DailySchedule] = None
    saturday: Optional[DailySchedule] = None
    sunday: Optional[DailySchedule] = None
    total_weekly_duration: str  # HH:MM:SS format
    pillar_distribution: Dict[str, float]  # percentage of time per pillar

class WeeklyCompletionStatus(BaseModel):
    week_start: datetime
    is_complete: bool = False

class ActionItem(BaseModel):
    title: str
    description: str
    priority: ActionPriority
    time_estimate: TimeEstimate
    prerequisites: Optional[List[str]] = None
    success_criteria: List[str]
    adaptation_notes: Optional[List[str]] = None  # Health-specific adaptations
    weekly_schedule: Optional[WeeklyActionSchedule] = None  # New field
    weekly_completion: List[WeeklyCompletionStatus] = Field(default_factory=list, description="Completion status for each week")

class ActionPlan(BaseModel):
    goal_id: str
    goal_title: str
    action_items: List[ActionItem]
    total_estimated_time_per_week: timedelta
    suggested_schedule: Optional[dict] = None  # Flexible schedule suggestions
    health_adaptations: List[str]  # Health-specific modifications to consider
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PlannerResponse(BaseModel):
    success: bool
    message: str
    data: Optional[ActionPlan] = None
