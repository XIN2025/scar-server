from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from .goals import Goal
from ..backend.diagnosis import DiagnosisResult

class HealthContext(BaseModel):
    goal_related_entities: List[str]
    medical_context: List[str]
    lifestyle_factors: List[str]
    risk_factors: List[str]

class HealthInsight(BaseModel):
    goal_id: str
    goal_title: str
    context: HealthContext
    diagnosis_summary: Optional[str]
    health_considerations: List[str]  # Health-specific factors to consider
    medical_precautions: List[str]  # Any medical warnings or precautions
    risk_factors: List[str]  # Identified risk factors related to the goal
    safety_notes: List[str]  # General safety considerations
    created_at: datetime = Field(default_factory=datetime.utcnow)

class HealthInsightResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None
