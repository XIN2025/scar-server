from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, time, timedelta
from enum import Enum

class TimeSlot(BaseModel):
    start_time: time
    end_time: time
    duration: timedelta
    pillar: Optional[str] = None
    action_item: Optional[str] = None
    frequency: Optional[str] = None
    priority: Optional[str] = None
    health_notes: Optional[List[str]] = None

class DaySchedule(BaseModel):
    date: datetime
    time_slots: List[TimeSlot]
    total_duration: timedelta
    pillars_covered: List[str]

class WeeklySchedule(BaseModel):
    start_date: datetime
    end_date: datetime
    daily_schedules: Dict[str, DaySchedule]  # key: day of week
    total_weekly_hours: float
    pillar_distribution: Dict[str, float]  # percentage of time per pillar
    health_adaptations: List[str]
    schedule_notes: Optional[List[str]] = None

class SchedulerResponse(BaseModel):
    success: bool
    message: str
    data: Optional[WeeklySchedule] = None
