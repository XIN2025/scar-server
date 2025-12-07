from typing import List, Optional
from ...schemas.ai.health_insights import HealthContext, HealthInsight
from ...schemas.ai.goals import Goal
from ...schemas.backend.diagnosis import DiagnosisRequest, DiagnosisResult
from .graph_db import get_graph_db
from .mai_dxo_service import MAIDxOService
import logging

logger = logging.getLogger(__name__)

class HealthInsightsService:
    def __init__(self):
        self.graph_db = get_graph_db()
        self.mai_dxo_service = MAIDxOService()

    def extract_goal_context(self, goal: Goal) -> HealthContext:
        """Extract relevant health context from the graph database for a given goal using semantic search"""
        try:
            query = f"Find health context for: {goal.title}. Details: {goal.description}"
            context_data = self.graph_db.get_context(
                query=query,
                user_email=goal.user_email
            )
            goal_entities = []
            medical_context = []
            lifestyle_factors = []
            risk_factors = []

            for item in context_data:
                if "Medical" in item or "Condition" in item or "Symptom" in item:
                    medical_context.append(item)
                elif "Lifestyle" in item or "Activity" in item:
                    lifestyle_factors.append(item)
                elif "Risk" in item or "Warning" in item:
                    risk_factors.append(item)
                else:
                    goal_entities.append(item)

            return HealthContext(
                goal_related_entities=goal_entities,
                medical_context=medical_context,
                lifestyle_factors=lifestyle_factors,
                risk_factors=risk_factors
            )
        except Exception as e:
            logger.error(f"Error extracting goal context: {str(e)}")
            raise

    def get_health_insight(self, goal: Goal, context: List[str] = None, user_email: str = None) -> HealthInsight:
        """Generate health insights for a given goal with optional context"""
        try:
            # If context is provided, create HealthContext from it
            if context:
                goal_entities = []
                medical_context = []
                lifestyle_factors = []
                risk_factors = []

                for item in context:
                    if "Medical" in item or "Condition" in item or "Symptom" in item:
                        medical_context.append(item)
                    elif "Lifestyle" in item or "Activity" in item:
                        lifestyle_factors.append(item)
                    elif "Risk" in item or "Warning" in item:
                        risk_factors.append(item)
                    else:
                        goal_entities.append(item)

                context = HealthContext(
                    goal_related_entities=goal_entities,
                    medical_context=medical_context,
                    lifestyle_factors=lifestyle_factors,
                    risk_factors=risk_factors
                )
            else:
                # Extract context from graph database if not provided
                context = self.extract_goal_context(goal)

            # Use provided user_email or fall back to goal.user_email
            user_email = user_email or goal.user_email

            diagnosis_request = DiagnosisRequest(
                user_email=user_email,  # Use the determined user_email
                symptoms=context.medical_context if context.medical_context else ["No specific symptoms noted"],
                contextual_data={  # Changed from context_data to contextual_data to match schema
                    "goal": {
                        "title": goal.title,
                        "description": goal.description
                    },
                    "medical_context": context.medical_context,
                    "lifestyle_factors": context.lifestyle_factors,
                    "risk_factors": context.risk_factors
                }
            )
            diagnosis_result = self.mai_dxo_service.generate_diagnosis(diagnosis_request)
            if hasattr(diagnosis_result, '__await__'):  # Handle if it's still a coroutine
                logger.error("mai_dxo_service.generate_diagnosis returned a coroutine, but it should be synchronous")
                diagnosis_result = None

            health_considerations = self._extract_health_considerations(diagnosis_result, context)
            medical_precautions = self._extract_medical_precautions(diagnosis_result, context)
            risk_factors = self._extract_risk_factors(diagnosis_result, context)
            safety_notes = self._generate_safety_notes(diagnosis_result, context)

            return HealthInsight(
                goal_id=str(goal.id),
                goal_title=goal.title,
                context=context,
                diagnosis_summary=diagnosis_result.summary if diagnosis_result else None,
                health_considerations=health_considerations,
                medical_precautions=medical_precautions,
                risk_factors=risk_factors,
                safety_notes=safety_notes
            )
        except Exception as e:
            logger.error(f"Error generating health insight: {str(e)}")
            raise



    def _extract_health_considerations(self, diagnosis: Optional[DiagnosisResult], context: HealthContext) -> List[str]:
        """Extract health considerations from diagnosis and context"""
        considerations = []
        
        if diagnosis and diagnosis.hypotheses:
            considerations.extend([f"Consider {h.diagnosis}: {h.justification}" for h in diagnosis.hypotheses])
            
            if diagnosis.checklist_feedback and diagnosis.checklist_feedback.recommendations:
                considerations.extend(diagnosis.checklist_feedback.recommendations)
        
        considerations.extend([ctx for ctx in context.medical_context if ctx not in considerations])
        
        return considerations

    def _extract_medical_precautions(self, diagnosis: Optional[DiagnosisResult], context: HealthContext) -> List[str]:
        """Extract medical precautions from diagnosis and context"""
        precautions = []
        
        if diagnosis and diagnosis.challenger_feedback:
            precautions.extend(diagnosis.challenger_feedback.missing_considerations)
            precautions.extend(diagnosis.challenger_feedback.risk_factors)
            
        precautions.extend([risk for risk in context.risk_factors if risk not in precautions])
        
        return precautions

    def _extract_risk_factors(self, diagnosis: Optional[DiagnosisResult], context: HealthContext) -> List[str]:
        """Extract risk factors from diagnosis and context"""
        return context.risk_factors

    def _generate_safety_notes(self, diagnosis: Optional[DiagnosisResult], context: HealthContext) -> List[str]:
        """Generate safety notes based on diagnosis and context"""
        safety_notes = []
        
        if diagnosis:
            for hypothesis in diagnosis.hypotheses:
                if hypothesis.confidence > 70:
                    safety_notes.append(f"Important - {hypothesis.diagnosis}: {hypothesis.justification}")
            
            if diagnosis.checklist_feedback:
                safety_notes.extend([f"Attention needed: {element}" for element in diagnosis.checklist_feedback.missing_elements])
            
            if diagnosis.challenger_feedback:
                safety_notes.append(f"Medical advisory: {diagnosis.challenger_feedback.critique}")
        
        for factor in context.lifestyle_factors:
            if any(kw in factor.lower() for kw in ['caution', 'warning', 'avoid', 'risk', 'safety']):
                safety_notes.append(factor)
                
        return safety_notes

health_insights_service = None

def get_health_insights_service() -> HealthInsightsService:
    global health_insights_service
    if health_insights_service is None:
        health_insights_service = HealthInsightsService()
    return health_insights_service
