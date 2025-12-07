from pydantic import BaseModel, Field
from app.schemas.ai.goals import StreakScore
from app.schemas.backend.health_alert import HealthDataScore
from app.schemas.ai.lab_report import LabReportScore
from typing import List, Optional

class HealthScore(BaseModel):
    score: float
    health_data_score: Optional[HealthDataScore] = None
    lab_report_score: List[LabReportScore] = Field(default_factory=list)
    streak_score: Optional[StreakScore] = None