from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class DiagnosisRequest(BaseModel):
    user_email: str
    symptoms: List[str]
    vital_signs: Optional[Dict[str, float]] = None
    medical_history: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    lab_results: Optional[Dict[str, str]] = None
    contextual_data: Optional[Dict[str, Any]] = None

class DiagnosisHypothesis(BaseModel):
    diagnosis: str
    confidence: float = Field(..., ge=0, le=100)
    justification: str

class ChallengerFeedback(BaseModel):
    critique: str
    missing_considerations: List[str]
    risk_factors: List[str]

class ChecklistFeedback(BaseModel):
    completeness_score: float = Field(..., ge=0, le=100)
    missing_elements: List[str]
    recommendations: List[str]

class DiagnosisResult(BaseModel):
    hypotheses: List[DiagnosisHypothesis]
    challenger_feedback: Optional[ChallengerFeedback]
    checklist_feedback: Optional[ChecklistFeedback]
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DiagnosisResponse(BaseModel):
    success: bool
    message: str
    data: Optional[DiagnosisResult] = None
