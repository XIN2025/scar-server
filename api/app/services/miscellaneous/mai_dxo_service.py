from datetime import datetime
from typing import List, Dict, Optional
from openai import OpenAI
from ...schemas.backend.diagnosis import (
    DiagnosisRequest,
    DiagnosisHypothesis,
    ChallengerFeedback,
    ChecklistFeedback,
    DiagnosisResult
)
from ..backend_services.db import get_db
from ...config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE
import logging

logger = logging.getLogger(__name__)
# 
class MAIDxOService:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
            
        self.db = get_db()
        self.diagnosis_collection = self.db["diagnoses"]
        
        # Initialize OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_rounds = 1  # Configuration for diagnostic rounds

    def _ask_model(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "insufficient_quota" in str(e):
                logger.error("OpenAI API quota exceeded. Please check your billing details.")
                raise ValueError("OpenAI API quota exceeded. Please check your billing details.")
            logger.error(f"OpenAI error: {str(e)}")
            raise  # Re-raise the exception to be handled by the caller

    def _format_case_summary(self, request: DiagnosisRequest) -> str:
        summary = [
            "CASE SUMMARY",
            "-------------",
            "Symptoms:",
            *[f"- {symptom}" for symptom in request.symptoms],
        ]

        if request.vital_signs:
            summary.extend([
                "\nVital Signs:",
                *[f"- {k}: {v}" for k, v in request.vital_signs.items()]
            ])

        if request.medical_history:
            summary.extend([
                "\nMedical History:",
                *[f"- {condition}" for condition in request.medical_history]
            ])

        if request.current_medications:
            summary.extend([
                "\nCurrent Medications:",
                *[f"- {med}" for med in request.current_medications]
            ])

        if request.lab_results:
            summary.extend([
                "\nLab Results:",
                *[f"- {test}: {result}" for test, result in request.lab_results.items()]
            ])

        if request.contextual_data:
            summary.extend([
                "\nContextual Information:",
                *[f"- {k}: {v}" for k, v in request.contextual_data.items()]
            ])

        return "\n".join(summary)

    def _parse_hypotheses(self, response: str) -> List[DiagnosisHypothesis]:
        try:
            hypotheses = []
            current_hypothesis = {
                'diagnosis': '',
                'confidence': 0.0,
                'justification': ''
            }
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('- Diagnosis:'):
                    if current_hypothesis['diagnosis']:  # Only add if we have a diagnosis
                        hypotheses.append(DiagnosisHypothesis(**current_hypothesis))
                        current_hypothesis = {
                            'diagnosis': '',
                            'confidence': 0.0,
                            'justification': ''
                        }
                    current_hypothesis['diagnosis'] = line.split(':', 1)[1].strip()
                elif line.startswith('  Confidence:'):
                    confidence = line.split(':', 1)[1].strip().rstrip('%')
                    current_hypothesis['confidence'] = float(confidence)
                elif line.startswith('  Justification:'):
                    current_hypothesis['justification'] = line.split(':', 1)[1].strip()

            if current_hypothesis['diagnosis']:  # Only add if we have a diagnosis
                hypotheses.append(DiagnosisHypothesis(**current_hypothesis))

            return hypotheses
        except Exception as e:
            logger.error(f"Error parsing hypotheses: {str(e)}")
            return []

    def _generate_challenger_feedback(self, case_summary: str, hypotheses: List[DiagnosisHypothesis]) -> ChallengerFeedback:
        prompt = (
            f"You are the ChallengerAgent. Review the following case and diagnoses critically:\n\n"
            f"Case Summary:\n{case_summary}\n\n"
            f"Current Hypotheses:\n"
            + "\n".join([f"- {h.diagnosis} ({h.confidence}%): {h.justification}" for h in hypotheses])
            + "\n\nProvide critical feedback on these diagnoses. Consider:\n"
            "1. What key factors might have been overlooked?\n"
            "2. Are there any inconsistencies or gaps in the reasoning?\n"
            "3. What risk factors should be evaluated?\n\n"
            "Format your response as:\n"
            "Critique: <main feedback>\n"
            "Missing Considerations: <list>\n"
            "Risk Factors: <list>"
        )

        try:
            response = self._ask_model(prompt)
            parts = response.split('\n')
            critique = next(p.split(':', 1)[1].strip() for p in parts if p.startswith('Critique:'))
            missing = next(p.split(':', 1)[1].strip() for p in parts if p.startswith('Missing Considerations:')).split(', ')
            risks = next(p.split(':', 1)[1].strip() for p in parts if p.startswith('Risk Factors:')).split(', ')

            return ChallengerFeedback(
                critique=critique,
                missing_considerations=missing,
                risk_factors=risks
            )
        except Exception as e:
            logger.error(f"Error parsing challenger feedback: {str(e)}")
            return ChallengerFeedback(
                critique="Error processing feedback",
                missing_considerations=[],
                risk_factors=[]
            )

    def _generate_checklist_feedback(self, case_summary: str, hypotheses: List[DiagnosisHypothesis]) -> ChecklistFeedback:
        prompt = (
            f"You are the ChecklistAgent. Review the following case and diagnoses for completeness:\n\n"
            f"Case Summary:\n{case_summary}\n\n"
            f"Current Hypotheses:\n"
            + "\n".join([f"- {h.diagnosis} ({h.confidence}%): {h.justification}" for h in hypotheses])
            + "\n\nEvaluate the diagnostic process for completeness. Consider:\n"
            "1. Have all relevant symptoms been considered?\n"
            "2. Are appropriate tests recommended?\n"
            "3. Is the diagnostic reasoning thorough?\n\n"
            "Format your response as:\n"
            "Completeness Score: <0-100>\n"
            "Missing Elements: <list>\n"
            "Recommendations: <list>"
        )
        

        try:
            response = self._ask_model(prompt)
            parts = response.split('\n')
            score = float(next(p.split(':', 1)[1].strip() for p in parts if p.startswith('Completeness Score:')))
            missing = next(p.split(':', 1)[1].strip() for p in parts if p.startswith('Missing Elements:')).split(', ')
            recommendations = next(p.split(':', 1)[1].strip() for p in parts if p.startswith('Recommendations:')).split(', ')

            return ChecklistFeedback(
                completeness_score=score,
                missing_elements=missing,
                recommendations=recommendations
            )
        except Exception as e:
            logger.error(f"Error parsing checklist feedback: {str(e)}")
            return ChecklistFeedback(
                completeness_score=0.0,
                missing_elements=[],
                recommendations=[]
            )

    def _run_diagnostic_round(self, case_summary: str, round_num: int, 
                               prev_hypotheses=None, prev_challenger=None, prev_checklist=None) -> tuple:
        """Run a single round of the diagnostic process with all agents."""
        
        # Generate or refine hypotheses
        prompt = (
            f"You are the HypothesisAgent in Round {round_num}. "
            f"{'Based on the following case summary, list and rank 5 possible diagnoses' if round_num == 1 else 'Update your diagnoses based on the feedback'}"
            f" with confidence scores and justifications.\n\n"
            f"Case Summary:\n{case_summary}\n\n"
        )

        if round_num > 1 and prev_challenger and prev_checklist:
            prompt += (
                f"Previous Hypotheses:\n" + 
                "\n".join([f"- {h.diagnosis} ({h.confidence}%): {h.justification}" for h in prev_hypotheses]) +
                f"\n\nChallenger Feedback:\n{prev_challenger.critique}\n" +
                f"Missing Considerations: {', '.join(prev_challenger.missing_considerations)}\n" +
                f"Risk Factors: {', '.join(prev_challenger.risk_factors)}\n\n" +
                f"Checklist Feedback:\n" +
                f"Completeness Score: {prev_checklist.completeness_score}\n" +
                f"Missing Elements: {', '.join(prev_checklist.missing_elements)}\n" +
                f"Recommendations: {', '.join(prev_checklist.recommendations)}\n\n"
            )

        prompt += (
            "If information is missing or ambiguous, make your best clinical guess based on the available data. "
            "Do not return an empty list. Always provide plausible hypotheses, even if you are uncertain.\n"
            "Format:\n- Diagnosis: <name>\n  Confidence: <score>%\n  Justification: <reason>"
        )

        try:
            response = self._ask_model(prompt)
            hypotheses = self._parse_hypotheses(response)

            # Generate feedback from both agents
            challenger_feedback = self._generate_challenger_feedback(case_summary, hypotheses)
            checklist_feedback = self._generate_checklist_feedback(case_summary, hypotheses)

            return hypotheses, challenger_feedback, checklist_feedback
        except ValueError as e:
            if "quota exceeded" in str(e):
                raise
            logger.error(f"Error in diagnostic round: {str(e)}")
            return [], None, None

    def _evaluate_convergence(self, current_hypotheses: List[DiagnosisHypothesis], 
                            previous_hypotheses: List[DiagnosisHypothesis]) -> bool:
        """Check if the diagnostic process has converged."""
        if not previous_hypotheses:
            return False

        # Compare top diagnoses and their confidence scores
        current_top = sorted(current_hypotheses, key=lambda x: x.confidence, reverse=True)[:3]
        previous_top = sorted(previous_hypotheses, key=lambda x: x.confidence, reverse=True)[:3]

        # Check if top diagnoses are the same and confidence scores haven't changed significantly
        confidence_threshold = 5.0  # 5% change in confidence is considered significant
        diagnoses_match = all(c.diagnosis == p.diagnosis for c, p in zip(current_top, previous_top))
        confidence_stable = all(abs(c.confidence - p.confidence) < confidence_threshold 
                              for c, p in zip(current_top, previous_top))

        return diagnoses_match and confidence_stable

    def generate_diagnosis(self, request: DiagnosisRequest) -> DiagnosisResult:
        """Run the complete diagnostic orchestration process."""
        case_summary = self._format_case_summary(request)
        max_rounds = self.max_rounds
        
        # Initialize tracking variables
        current_round = 1
        prev_hypotheses = None
        prev_challenger = None
        prev_checklist = None
        final_hypotheses = None
        final_challenger = None
        final_checklist = None

        while current_round <= max_rounds:
            logger.info(f"Starting diagnostic round {current_round}")
            
            # Run the current round
            hypotheses, challenger_feedback, checklist_feedback = self._run_diagnostic_round(
                case_summary, current_round, prev_hypotheses, prev_challenger, prev_checklist
            )

            # Check for convergence if not the first round
            if current_round > 1 and self._evaluate_convergence(hypotheses, prev_hypotheses):
                logger.info(f"Diagnostic process converged after {current_round} rounds")
                break

            # Update for next round
            prev_hypotheses = hypotheses
            prev_challenger = challenger_feedback
            prev_checklist = checklist_feedback
            
            # Store the final state
            final_hypotheses = hypotheses
            final_challenger = challenger_feedback
            final_checklist = checklist_feedback

            current_round += 1

        # Generate a summary
        summary = ""
        if final_hypotheses:
            # Get top hypotheses by confidence
            top_hypotheses = sorted(final_hypotheses, key=lambda x: x.confidence, reverse=True)[:3]
            summary_parts = [
                f"{h.diagnosis} ({h.confidence}% confidence): {h.justification}"
                for h in top_hypotheses
            ]
            summary = "Primary findings: " + "; ".join(summary_parts)
            
            if final_challenger and final_challenger.critique:
                summary += f"\nKey considerations: {final_challenger.critique}"
            
            if final_checklist and final_checklist.recommendations:
                summary += f"\nRecommended actions: {'; '.join(final_checklist.recommendations)}"

        # Create the final result
        result = DiagnosisResult(
            hypotheses=final_hypotheses,
            challenger_feedback=final_challenger,
            checklist_feedback=final_checklist,
            summary=summary
        )

        # Store in database with all context
        self.diagnosis_collection.insert_one({
            "user_email": request.user_email,
            "request": request.dict(),
            "result": result.dict(),
            "rounds_completed": current_round - 1,
            "converged": current_round < max_rounds,
            "created_at": datetime.utcnow()
        })

        return result
