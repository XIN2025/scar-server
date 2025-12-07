import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import logging
from datetime import datetime, date

from app.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from app.services.ai_services.mongodb_vectorstore import get_vector_store
from app.utils.ai.prompts import ChatPrompts
from app.services.backend_services.db import get_db
from app.services.ai_services.lab_report_service import get_lab_report_service
from app.services.backend_services.health_alert_service import get_health_alert_service
from app.services.ai_services.memory_service import get_memory_service
from app.services.ai_services.personalization_profile_service import get_personalization_profile_service

logger = logging.getLogger(__name__)

@dataclass
class ChatState:
    query: str
    user_email: str
    context: List[str]
    response: str = ""
    follow_up_questions: List[str] = field(default_factory=list)
    reasoning: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    is_health_related: bool = True
    memories: List[Dict[str, Any]] = None
    health_type: str = "NONE"  # SYMPTOM | WELLNESS | INFORMATIONAL | NONE
    medical_risk_level: str = "LOW"  # LOW | MODERATE | HIGH
    personalization_profile: Optional[Dict[str, Any]] = None  


class ChatService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=float(LLM_TEMPERATURE),
        )
        self.vector_store = get_vector_store()
        self.db = get_db()
        self.user_collection = self.db["users"]
        self.preferences_collection = self.db["preferences"]
        self.lab_report_service = get_lab_report_service()
        self.health_alert_service = get_health_alert_service()
        self.memory_service = get_memory_service()
        self.personalization_service = get_personalization_profile_service()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ChatState)
        workflow.add_node("health_relevance_check", self._health_relevance_check_node)
        workflow.add_node("context_retrieval", self._context_retrieval_node)
        workflow.add_node("response_generation", self._response_generation_node)
        workflow.add_node("follow_up_generation", self._follow_up_generation_node)
        workflow.set_entry_point("health_relevance_check")
        
        workflow.add_conditional_edges(
            "health_relevance_check",
            self._route_after_health_check,
            {
                "health_related": "context_retrieval",
                "not_health_related": "response_generation"
            }
        )
        
        workflow.add_edge("context_retrieval", "response_generation")
        workflow.add_edge("response_generation", "follow_up_generation")
        workflow.add_edge("follow_up_generation", END)
        return workflow.compile()
    
    def _route_after_health_check(self, state: ChatState) -> str:
        """Route based on health relevance classification"""
        if isinstance(state, dict): 
            state = ChatState(**state)
        
        # Use the boolean flag for routing
        if not state.is_health_related:
            return "not_health_related"
        else:
            return "health_related"

    async def _health_relevance_check_node(self, state: ChatState) -> dict:
        if isinstance(state, dict):
            state = ChatState(**state)

        logger.info(f"🛡️ [GUARDRAILS] Advanced health intent analysis for: '{state.query}'")

        classification_prompt = f"""
    You are a healthcare intent classifier for a clinical health assistant named Scar.

    Analyze the user's query and return JSON with the following structure:

    {{
    "intent": "HEALTH" | "NOT_HEALTH",
    "health_type": "SYMPTOM" | "WELLNESS" | "INFORMATIONAL" | "NONE",
    "medical_risk_level": "LOW" | "MODERATE" | "HIGH"
    }}

    Definitions:

    HEALTH intent includes:
    - Symptoms, illness, pain, discomfort
    - Mental health concerns
    - Sleep, fatigue, stress, lifestyle changes
    - Fitness, diet, wellness improvement
    - Questions about medical conditions or prevention

    NOT_HEALTH includes:
    - Sports, celebrities, trivia, current events, politics, general knowledge

    health_type meanings:
    - SYMPTOM → user describing how they feel physically or mentally
    - WELLNESS → lifestyle improvement
    - INFORMATIONAL → medical curiosity
    - NONE → not health related

    medical_risk_level:
    - HIGH → severe symptoms, urgent risk (chest pain, breathlessness, fainting)
    - MODERATE → concerning but not critical
    - LOW → general or mild concerns

    User query:
    "{state.query}"

    Respond ONLY with valid JSON, no explanation.
    """

        try:
            response = self.llm.invoke([HumanMessage(content=classification_prompt)])
            import json
            result = json.loads(response.content.strip())

            intent = result.get("intent")
            state.health_type = result.get("health_type", "NONE")
            state.medical_risk_level = result.get("medical_risk_level", "LOW")

            if intent == "HEALTH":
                state.is_health_related = True
                state.reasoning = f"Health intent detected ({state.health_type}, risk: {state.medical_risk_level})"
                return asdict(state)

            state.is_health_related = False
            state.response = (
                "I'm Scar, your health companion 🩺 I focus on health, medical, sleep, "
                "fitness and wellbeing topics. For general information like sports or trivia, "
                "please use a general-purpose assistant."
            )
            state.context = []
            state.follow_up_questions = []
            state.reasoning = "Non-health intent detected"
            return asdict(state)

        except Exception as e:
            logger.error(f"❌ Guardrail parsing failure: {e}")
            state.is_health_related = True
            state.health_type = "INFORMATIONAL"
            state.medical_risk_level = "LOW"
            state.reasoning = "Classifier fallback"
            return asdict(state)

    async def _context_retrieval_node(self, state: ChatState) -> dict:
        if isinstance(state, dict): state = ChatState(**state)
        logger.info(f"🔍 [CONTEXT RETRIEVAL] Starting context retrieval for query: '{state.query}'")
        
        # Fetch all context in parallel
        async def fetch_vector_context():
            try:
                # Vector store search is synchronous, run it in thread pool to avoid blocking
                relevant_docs = await asyncio.to_thread(
                    self.vector_store.search,
                    state.query,
                    state.user_email,
                    10
                )
                logger.info(f"🔍 [CONTEXT RETRIEVAL] Retrieved {len(relevant_docs)} docs from vector store")
                context_pieces = [doc.get("text", "") for doc in relevant_docs if doc.get("text", "").strip()]
                return context_pieces
            except Exception as e:
                logger.error(f"❌ [CONTEXT RETRIEVAL] Error during vector context retrieval: {e}", exc_info=True)
                return []
        
        async def fetch_user():
            try:
                user = await self.user_collection.find_one({"email": state.user_email})
                return user
            except Exception as e:
                logger.error(f"❌ [CONTEXT RETRIEVAL] Error fetching user data: {e}")
                return None
        
        async def fetch_intake_form():
            try:
                intake = await self.preferences_collection.find_one({"email": state.user_email})
                if not intake:
                    return None
                return intake
            except Exception as e:
                logger.error(f"❌ [CONTEXT RETRIEVAL] Error fetching intake form: {e}")
                return None
        
        async def fetch_lab_reports():
            try:
                reports = await self.lab_report_service.get_lab_reports_by_user(state.user_email)
                if not reports:
                    return []
                # Get detailed reports for summary (limit to 5 most recent) - fetch in parallel
                report_tasks = [
                    self.lab_report_service.get_lab_report_by_id(report_summary.id, state.user_email)
                    for report_summary in reports[:5]
                ]
                detailed_results = await asyncio.gather(*report_tasks, return_exceptions=True)
                detailed_reports = []
                for result in detailed_results:
                    if isinstance(result, Exception):
                        logger.warning(f"Error fetching detailed lab report: {result}")
                        continue
                    if result:
                        detailed_reports.append(result)
                return detailed_reports
            except Exception as e:
                logger.error(f"❌ [CONTEXT RETRIEVAL] Error fetching lab reports: {e}")
                return []
        
        async def fetch_health_data():
            try:
                health_data = await self.health_alert_service.get_latest_health_data_by_user_email(state.user_email)
                return health_data
            except Exception as e:
                logger.error(f"❌ [CONTEXT RETRIEVAL] Error fetching health data: {e}")
                return None
        
        async def fetch_personalization_profile():
            """Fetch or generate personalization profile if needed."""
            try:
                should_generate, reason = await self.personalization_service.should_generate_profile(state.user_email)
                
                if should_generate:
                    logger.info(f"🎯 [PERSONALIZATION] Generating new profile: {reason}")
                    try:
                        profile = await self.personalization_service.generate_profile(state.user_email)
                        logger.info(f"✅ [PERSONALIZATION] Profile generated successfully")
                        return profile
                    except Exception as e:
                        logger.warning(f"⚠️ [PERSONALIZATION] Failed to generate profile: {e}")
                        return None
                else:
                    profile = await self.personalization_service.get_cached_profile(state.user_email)
                    if profile:
                        logger.info(f"✅ [PERSONALIZATION] Using cached profile (reason: {reason})")
                    else:
                        logger.info(f"ℹ️ [PERSONALIZATION] No profile available (reason: {reason})")
                    return profile
            except Exception as e:
                logger.warning(f"⚠️ [PERSONALIZATION] Error handling profile: {e}")
                return None
        
        # Fetch all in parallel
        results = await asyncio.gather(
            fetch_vector_context(),
            fetch_user(),
            fetch_intake_form(),
            fetch_lab_reports(),
            fetch_health_data(),
            fetch_personalization_profile(),
            return_exceptions=True
        )
        
        # Unpack results with error handling
        context_pieces = results[0] if not isinstance(results[0], Exception) else []
        user_data = results[1] if not isinstance(results[1], Exception) else None
        intake_data = results[2] if not isinstance(results[2], Exception) else None
        lab_reports = results[3] if not isinstance(results[3], Exception) else []
        health_data = results[4] if not isinstance(results[4], Exception) else None
        personalization_profile = results[5] if not isinstance(results[5], Exception) else None
        
        # Store context in state
        if context_pieces:
            logger.info(f"✅ [CONTEXT RETRIEVAL] Extracted {len(context_pieces)} non-empty context pieces.")
        else:
            logger.warning("⚠️ [CONTEXT RETRIEVAL] No context found after search.")
        
        state.context = context_pieces
        state.reasoning = f"Retrieved {len(context_pieces)} relevant text chunks from vector store."
        
        # Store additional data in state for use in response generation
        if user_data:
            state._user_data = user_data
        if intake_data:
            state._intake_data = intake_data
        if lab_reports:
            state._lab_reports = lab_reports
        if health_data:
            state._health_data = health_data
        if personalization_profile:
            state.personalization_profile = personalization_profile
            logger.info(f"✅ [PERSONALIZATION] Profile added to state")
        
        # Convert to dict and manually add underscore-prefixed attributes (asdict doesn't include them)
        result_dict = asdict(state)
        if hasattr(state, '_user_data'):
            result_dict['_user_data'] = state._user_data
        if hasattr(state, '_intake_data'):
            result_dict['_intake_data'] = state._intake_data
        if hasattr(state, '_lab_reports'):
            result_dict['_lab_reports'] = state._lab_reports
        if hasattr(state, '_health_data'):
            result_dict['_health_data'] = state._health_data
        
        logger.info(f"✅ [CONTEXT RETRIEVAL] Retrieved vector docs + intake form + {len(lab_reports)} lab reports + health data")
        return result_dict

    async def _response_generation_node(self, state: ChatState) -> dict:
        # Handle dict input and preserve underscore attributes
        if isinstance(state, dict):
            # Preserve underscore attributes before converting
            intake_data = state.get('_intake_data')
            lab_reports = state.get('_lab_reports', [])
            health_data = state.get('_health_data')
            user_data = state.get('_user_data')
            
            state = ChatState(**{k: v for k, v in state.items() if not k.startswith('_')})
            
            # Restore underscore attributes
            if intake_data:
                state._intake_data = intake_data
            if lab_reports:
                state._lab_reports = lab_reports
            if health_data:
                state._health_data = health_data
            if user_data:
                state._user_data = user_data
        
        if not state.is_health_related and state.response:
            logger.info(f"💬 [RESPONSE GENERATION] Non-health query with template response already set, skipping generation: '{state.query}'")
            return asdict(state)
        
        logger.info(f"💬 [RESPONSE GENERATION] Starting response generation for query: '{state.query}'")
        logger.info(f"💬 [RESPONSE GENERATION] Context pieces available: {len(state.context)}")

        user_context = "Patient details are not available."
        # Use pre-fetched user data if available (from context retrieval node)
        user = getattr(state, '_user_data', None)
        
        if not user:
            # Fallback: fetch user if not already fetched
            try:
                user = await self.user_collection.find_one({"email": state.user_email})
            except Exception as e:
                logger.error(f"❌ [RESPONSE GENERATION] Error fetching user data for {state.user_email}: {e}")
                user = None
        
        if user:
            def calculate_age(dob_str):
                if not dob_str: return "unknown"
                try:
                    dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
                    today = date.today()
                    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                except (ValueError, TypeError): return "unknown"
            
            user_name = user.get("name", "User")
            date_of_birth = user.get("date_of_birth")
            age = calculate_age(date_of_birth)
            blood_type = user.get("blood_type", "unknown")
            user_context = f"Patient's name is {user_name}. Age is {age}. Blood type is {blood_type}."
            logger.info(f"💬 [RESPONSE GENERATION] Found user context: {user_context}")
        else:
            logger.warning(f"⚠️ [RESPONSE GENERATION] User with email {state.user_email} not found in 'users' collection.")

        # Build additional context sections
        intake_form_context = ""
        lab_report_summary = ""
        health_data_context = ""
        
        # Format intake form data
        intake_data = getattr(state, '_intake_data', None)
        if intake_data:
            intake_parts = []
            if intake_data.get("age"):
                intake_parts.append(f"Age: {intake_data['age']}")
            if intake_data.get("gender"):
                intake_parts.append(f"Gender: {intake_data['gender']}")
            if intake_data.get("healthGoals"):
                goals = ", ".join(intake_data.get("healthGoals", []))
                intake_parts.append(f"Health Goals: {goals}")
            if intake_data.get("conditions"):
                conditions = ", ".join(intake_data.get("conditions", []))
                intake_parts.append(f"Current Conditions: {conditions}")
            if intake_data.get("atRiskConditions"):
                at_risk = ", ".join(intake_data.get("atRiskConditions", []))
                intake_parts.append(f"At-Risk Conditions: {at_risk}")
            if intake_data.get("communicationStyle"):
                intake_parts.append(f"Communication Style: {intake_data['communicationStyle']}")
            
            if intake_parts:
                intake_form_context = "\n".join(intake_parts)
        
        # Generate lab report summary
        lab_reports = getattr(state, '_lab_reports', [])
        if lab_reports:
            try:
                # Build summary of lab reports
                reports_text = []
                for report in lab_reports:
                    report_info = f"Test: {report.test_title}\n"
                    if report.test_date:
                        report_info += f"Date: {report.test_date.strftime('%Y-%m-%d')}\n"
                    if report.test_description:
                        report_info += f"Description: {report.test_description}\n"
                    report_info += "Key Results:\n"
                    
                    # Include notable results (abnormal or important values)
                    notable_props = []
                    properties = getattr(report, 'properties', []) or []
                    for prop in properties[:15]:
                        status = getattr(prop, 'status', None)
                        name = getattr(prop, 'property_name', 'Unknown')
                        value = getattr(prop, 'value', 'N/A')
                        if status and str(status).lower() in ['high', 'low', 'abnormal', 'critical']:
                            notable_props.append(f"  - {name}: {value} ({status})")
                        elif len(notable_props) < 5:  # Include some normal values too
                            notable_props.append(f"  - {name}: {value}")
                    
                    report_info += "\n".join(notable_props[:10])  # Limit to 10 results per report
                    reports_text.append(report_info)
                
                if reports_text:
                    # Generate AI summary
                    summary_prompt = f"""Summarize the following lab reports in a concise format (2-3 paragraphs max) that highlights:
1. Key findings and any abnormalities
2. Overall health trends
3. Areas that may need attention

Lab Reports:
{chr(10).join(reports_text)}

Provide a concise, actionable summary focusing on the most important insights."""
                    
                    summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
                    lab_report_summary = summary_response.content.strip()
            except Exception as e:
                logger.error(f"❌ [RESPONSE GENERATION] Error generating lab report summary: {e}")
                # Fallback to basic summary
                lab_report_summary = f"Lab Reports Available: {len(lab_reports)} recent test(s)."
        
        # Format health data
        health_data = getattr(state, '_health_data', None)
        if health_data:
            health_parts = []
            summary = health_data.aggregated_summary
            
            if summary.step and summary.step.total:
                health_parts.append(f"Total Steps: {summary.step.total}")
            
            if summary.heartRate:
                health_parts.append(f"Heart Rate: Avg {summary.heartRate.average:.0f} bpm (Max: {summary.heartRate.max:.0f}, Min: {summary.heartRate.min:.0f})")
            
            if summary.activeEnergy and summary.activeEnergy.total:
                health_parts.append(f"Active Energy: {summary.activeEnergy.total:.0f} kcal")
            
            if summary.sleep and summary.sleep.totalHours:
                health_parts.append(f"Sleep: {summary.sleep.totalHours:.2f} hours")
            
            if summary.weight and summary.weight.value:
                health_parts.append(f"Weight: {summary.weight.value:.2f} {summary.weight.unit}")
            
            if summary.bloodGlucose and summary.bloodGlucose.average:
                health_parts.append(f"Blood Glucose: Avg {summary.bloodGlucose.average:.2f} mg/dL")
            
            if summary.oxygenSaturation and summary.oxygenSaturation.average:
                health_parts.append(f"Oxygen Saturation: Avg {summary.oxygenSaturation.average:.1f}%")
            
            # Include recent hourly data (last 3-5 hours)
            if health_data.hourly_data:
                recent_hours = health_data.hourly_data[-5:]
                health_parts.append(f"\nRecent Hourly Data (Last {len(recent_hours)} hours):")
                for hour_data in recent_hours:
                    hour_time = hour_data.created_at.strftime('%H:%M') if hour_data.created_at else 'N/A'
                    hour_metrics = []
                    steps_val = getattr(getattr(getattr(hour_data, 'data', None), 'steps', None), 'value', None)
                    if steps_val is not None:
                        hour_metrics.append(f"Steps: {steps_val}")
                    hr_val = getattr(getattr(getattr(hour_data, 'data', None), 'heartRate', None), 'value', None)
                    if hr_val is not None:
                        hour_metrics.append(f"HR: {hr_val}")
                    sleep_val = getattr(getattr(getattr(hour_data, 'data', None), 'sleep', None), 'value', None)
                    if isinstance(sleep_val, (int, float)):
                        hour_metrics.append(f"Sleep: {sleep_val:.2f}h")
                    elif sleep_val is not None:
                        hour_metrics.append(f"Sleep: {sleep_val}h")
                    if hour_metrics:
                        health_parts.append(f"  [{hour_time}] {', '.join(hour_metrics)}")
            
            if health_parts:
                health_data_context = "\n".join(health_parts)

        # Format memories for prompt
        memories = getattr(state, 'memories', []) or []
        
        personalization_context = ""
        if state.personalization_profile:
            try:
                personalization_context = self.personalization_service.format_profile_for_chat(
                    state.personalization_profile
                )
                if personalization_context:
                    logger.info(f"✅ [PERSONALIZATION] Added profile context to prompt ({len(personalization_context)} chars)")
            except Exception as e:
                logger.warning(f"⚠️ [PERSONALIZATION] Error formatting profile: {e}")
        
        try:
            rag_prompt = ChatPrompts.get_medical_rag_prompt(
                query=state.query,
                personal_info=user_context,
                medical_history=state.context,
                intake_form_context=intake_form_context,
                lab_report_summary=lab_report_summary,
                health_data_context=health_data_context,
                memories=memories
            )
            
            if personalization_context:
                rag_prompt += f"\n\n=== PERSONALIZED HEALTH PROFILE ===\n{personalization_context}\n\nUse this personalized profile to tailor your responses to the user's specific health goals, lifestyle, and therapeutic focus areas.\n"
            
            system_safety_block = """
            You are Scar, a compassionate health companion.

            You MUST follow these medical communication principles:
            - Do not provide diagnoses unless explicitly requested.
            - Never present advice as a substitute for professional medical consultation.
            - Use empathetic, calm language.
            - If symptoms may indicate risk, recommend consulting a doctor.
            - For high risk, include urgency such as:
            "Please seek immediate medical attention."

            Tone:
            - Supportive
            - Non-judgmental
            - Reassuring but not dismissive
            - Clear and grounded
            """
            rag_prompt = system_safety_block + "\n\n" + rag_prompt
            # HIGH RISK ESCALATION
            if getattr(state, "medical_risk_level", None) == "HIGH":
                rag_prompt += """
            IMPORTANT SAFETY NOTE:
            This user's symptoms may indicate a serious medical concern.
            Strongly advise urgent medical evaluation or emergency services.
            """

            # SYMPTOM BEHAVIOUR
            if getattr(state, "health_type", None) == "SYMPTOM":
                rag_prompt += """
            When responding, ask structured follow-up questions:
            - Duration of symptoms
            - Severity
            - Triggers
            - Associated symptoms
            """
            logger.info("💬 [RESPONSE GENERATION] Sending prompt to LLM...")
            
            response = self.llm.invoke([HumanMessage(content=rag_prompt)])
            state.response = response.content.strip()
            logger.info("💬 [RESPONSE GENERATION] ✅ LLM response received successfully.")
        except Exception as e:
            logger.error(f"❌ [RESPONSE GENERATION] LLM call failed: {e}", exc_info=True)
            state.response = "I’m sorry, but I encountered an error while processing your request. Please try again later."
        
        return asdict(state)

    async def _follow_up_generation_node(self, state: ChatState) -> dict:
        if isinstance(state, dict): state = ChatState(**state)
        
        # Skip follow-ups for non-health responses or error responses
        if not state.is_health_related or not state.response or "sorry" in state.response.lower() or "error" in state.response.lower():
            state.follow_up_questions = []
            return asdict(state)

        follow_up_prompt = ChatPrompts.get_follow_up_questions_prompt(state.query, state.response)
        try:
            response = self.llm.invoke([HumanMessage(content=follow_up_prompt)])
            follow_up_questions = [q.strip().lstrip("- ").lstrip("* ") for q in response.content.strip().split('\n') if q.strip()]
            state.follow_up_questions = follow_up_questions[:4]
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            state.follow_up_questions = []
            
        return asdict(state)
    
    async def chat(self, query: str, user_email: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        logger.info(f"🚀 [CHAT] Starting new chat for user '{user_email}' with query: '{query}'")
        try:
            # Retrieve relevant memories for the user
            memories = []
            try:
                memories = await self.memory_service.get_memories(user_email, query, limit=5)
                logger.info(f"🧠 [MEMORY] Retrieved {len(memories)} relevant memories for user {user_email}")
            except Exception as e:
                logger.warning(f"⚠️ [MEMORY] Failed to retrieve memories: {e}")
            
            initial_state = ChatState(
                query=query, 
                user_email=user_email, 
                context=[], 
                follow_up_questions=[], 
                conversation_history=conversation_history or [],
                memories=memories
            )
            logger.info(f"🚀 [CHAT] Step 1: ✅ ChatState initialized: {initial_state}")
            
            logger.info("🚀 [CHAT] Step 2: Running LangGraph workflow...")
            result = await self.graph.ainvoke(initial_state)
            logger.info(f"🚀 [CHAT] Step 2: ✅ Workflow completed successfully!")
            
            # Store conversation in memory after successful completion
            if result.get("response"):
                try:
                    messages_to_store = []
                    if conversation_history:
                        messages_to_store.extend(conversation_history)
                    # Add current interaction
                    messages_to_store.append({"role": "user", "content": query})
                    messages_to_store.append({"role": "assistant", "content": result.get("response", "")})
                    
                    await self.memory_service.add_memory(user_email, messages_to_store)
                    logger.info(f"🧠 [MEMORY] Stored conversation in memory for user {user_email}")
                except Exception as e:
                    logger.warning(f"⚠️ [MEMORY] Failed to store memories: {e}")
            
            response_data = {
                "success": True,
                "response": result.get("response", ""),
                "follow_up_questions": result.get("follow_up_questions", []),
                "context_used": len(result.get("context", [])),
                "reasoning": result.get("reasoning", "")
            }
            logger.info(f"🚀 [CHAT] Step 3: ✅ Returning response: {response_data}")
            return response_data
            

        except Exception as e:
            logger.error(f"❌ [CHAT] Critical error in chat workflow for user '{user_email}': {e}", exc_info=True)
            return {
                "success": False,
                "response": "I apologize, but a critical error occurred. The technical team has been notified.",
                "follow_up_questions": [],
                "error": str(e)
            }

    async def chat_stream(self, query: str, user_email: str, conversation_history: List[Dict[str, str]] = None):
        try:
            # Retrieve relevant memories for the user
            memories = []
            try:
                memories = await self.memory_service.get_memories(user_email, query, limit=5)
                logger.info(f"🧠 [MEMORY] Retrieved {len(memories)} relevant memories for user {user_email}")
            except Exception as e:
                logger.warning(f"⚠️ [MEMORY] Failed to retrieve memories: {e}")
            
            logger.info(f"🛡️ [STREAM CHAT] Checking health relevance for query: '{query}'")
            health_check_state = ChatState(
                query=query, 
                user_email=user_email, 
                context=[], 
                follow_up_questions=[], 
                conversation_history=conversation_history or [],
                memories=memories
            )
            
            health_result = await self._health_relevance_check_node(health_check_state)
            health_type = health_result.get("health_type")
            medical_risk_level = health_result.get("medical_risk_level")
            if not health_result.get("is_health_related", False):
                logger.info("🛡️ [STREAM CHAT] Non-health query detected, returning template response")
                response_text = health_result.get("response", "I'm a health-focused assistant and can only help with health, medical, and wellness questions.")
                # Stream the response as a single chunk
                yield {"type": "response_chunk", "content": response_text}
                yield {"type": "follow_up", "content": []}
                return
            
            logger.info(f"🔍 [STREAM CHAT] Health-related query, starting context retrieval for query: '{query}'")
            context_state = ChatState(query=query, user_email=user_email, context=[], follow_up_questions=[], conversation_history=conversation_history or [])
            
            context_result = await self._context_retrieval_node(context_state)
            context_pieces = context_result.get("context", [])
            
            # Get additional context from state (already fetched in _context_retrieval_node)
            user_data = context_result.get("_user_data")
            intake_data = context_result.get("_intake_data")
            lab_reports = context_result.get("_lab_reports", [])
            health_data = context_result.get("_health_data")
            personalization_profile = context_result.get("personalization_profile")
            
            # Format user context from pre-fetched data
            user_context = "Patient details are not available."
            if user_data:
                try:
                    def calculate_age(dob_str):
                        if not dob_str: return "unknown"
                        try:
                            dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
                            today = date.today()
                            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                        except (ValueError, TypeError): return "unknown"
                    
                    user_name = user_data.get("name", "User")
                    date_of_birth = user_data.get("date_of_birth")
                    age = calculate_age(date_of_birth)
                    blood_type = user_data.get("blood_type", "unknown")
                    user_context = f"Patient's name is {user_name}. Age is {age}. Blood type is {blood_type}."
                except Exception as e:
                    logger.error(f"❌ [STREAM CHAT] Error formatting user data: {e}")
            
            # Format intake form data
            intake_form_context = ""
            if intake_data:
                intake_parts = []
                if intake_data.get("age"):
                    intake_parts.append(f"Age: {intake_data['age']}")
                if intake_data.get("gender"):
                    intake_parts.append(f"Gender: {intake_data['gender']}")
                if intake_data.get("healthGoals"):
                    goals = ", ".join(intake_data.get("healthGoals", []))
                    intake_parts.append(f"Health Goals: {goals}")
                if intake_data.get("conditions"):
                    conditions = ", ".join(intake_data.get("conditions", []))
                    intake_parts.append(f"Current Conditions: {conditions}")
                if intake_data.get("atRiskConditions"):
                    at_risk = ", ".join(intake_data.get("atRiskConditions", []))
                    intake_parts.append(f"At-Risk Conditions: {at_risk}")
                if intake_data.get("communicationStyle"):
                    intake_parts.append(f"Communication Style: {intake_data['communicationStyle']}")
                if intake_parts:
                    intake_form_context = "\n".join(intake_parts)
            
            # Generate lab report summary
            lab_report_summary = ""
            if lab_reports:
                try:
                    reports_text = []
                    for report in lab_reports:
                        report_info = f"Test: {report.test_title}\n"
                        if report.test_date:
                            report_info += f"Date: {report.test_date.strftime('%Y-%m-%d')}\n"
                        if report.test_description:
                            report_info += f"Description: {report.test_description}\n"
                        report_info += "Key Results:\n"
                        
                        notable_props = []
                        properties = getattr(report, 'properties', []) or []
                        for prop in properties[:15]:
                            status = getattr(prop, 'status', None)
                            name = getattr(prop, 'property_name', 'Unknown')
                            value = getattr(prop, 'value', 'N/A')
                            if status and str(status).lower() in ['high', 'low', 'abnormal', 'critical']:
                                notable_props.append(f"  - {name}: {value} ({status})")
                            elif len(notable_props) < 5:
                                notable_props.append(f"  - {name}: {value}")
                        
                        report_info += "\n".join(notable_props[:10])
                        reports_text.append(report_info)
                    
                    if reports_text:
                        summary_prompt = f"""Summarize the following lab reports in a concise format (2-3 paragraphs max) that highlights:
1. Key findings and any abnormalities
2. Overall health trends
3. Areas that may need attention

Lab Reports:
{chr(10).join(reports_text)}

Provide a concise, actionable summary focusing on the most important insights."""
                        
                        # Use async LLM call to avoid blocking
                        summary_response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
                        lab_report_summary = summary_response.content.strip()
                except Exception as e:
                    logger.error(f"❌ [STREAM CHAT] Error generating lab report summary: {e}")
                    lab_report_summary = f"Lab Reports Available: {len(lab_reports)} recent test(s)."
            
            # Format health data
            health_data_context = ""
            if health_data:
                health_parts = []
                summary = health_data.aggregated_summary
                
                if summary.step and summary.step.total:
                    health_parts.append(f"Total Steps: {summary.step.total}")
                if summary.heartRate:
                    health_parts.append(f"Heart Rate: Avg {summary.heartRate.average:.0f} bpm (Max: {summary.heartRate.max:.0f}, Min: {summary.heartRate.min:.0f})")
                if summary.activeEnergy and summary.activeEnergy.total:
                    health_parts.append(f"Active Energy: {summary.activeEnergy.total:.0f} kcal")
                if summary.sleep and summary.sleep.totalHours:
                    health_parts.append(f"Sleep: {summary.sleep.totalHours:.2f} hours")
                if summary.weight and summary.weight.value:
                    health_parts.append(f"Weight: {summary.weight.value:.2f} {summary.weight.unit}")
                if summary.bloodGlucose and summary.bloodGlucose.average:
                    health_parts.append(f"Blood Glucose: Avg {summary.bloodGlucose.average:.2f} mg/dL")
                if summary.oxygenSaturation and summary.oxygenSaturation.average:
                    health_parts.append(f"Oxygen Saturation: Avg {summary.oxygenSaturation.average:.1f}%")
                
                if health_data.hourly_data:
                    recent_hours = health_data.hourly_data[-5:]
                    health_parts.append(f"\nRecent Hourly Data (Last {len(recent_hours)} hours):")
                    for hour_data in recent_hours:
                        hour_time = hour_data.created_at.strftime('%H:%M') if hour_data.created_at else 'N/A'
                        hour_metrics = []
                        steps_val = getattr(getattr(getattr(hour_data, 'data', None), 'steps', None), 'value', None)
                        if steps_val is not None:
                            hour_metrics.append(f"Steps: {steps_val}")
                        hr_val = getattr(getattr(getattr(hour_data, 'data', None), 'heartRate', None), 'value', None)
                        if hr_val is not None:
                            hour_metrics.append(f"HR: {hr_val}")
                        sleep_val = getattr(getattr(getattr(hour_data, 'data', None), 'sleep', None), 'value', None)
                        if isinstance(sleep_val, (int, float)):
                            hour_metrics.append(f"Sleep: {sleep_val:.2f}h")
                        elif sleep_val is not None:
                            hour_metrics.append(f"Sleep: {sleep_val}h")
                        if hour_metrics:
                            health_parts.append(f"  [{hour_time}] {', '.join(hour_metrics)}")
                
                if health_parts:
                    health_data_context = "\n".join(health_parts)

            # Format memories for prompt
            memories = getattr(health_check_state, 'memories', []) or []
            
            personalization_context = ""
            if personalization_profile:
                try:
                    personalization_context = self.personalization_service.format_profile_for_chat(
                        personalization_profile
                    )
                    if personalization_context:
                        logger.info(f"✅ [STREAM PERSONALIZATION] Added profile context to prompt ({len(personalization_context)} chars)")
                except Exception as e:
                    logger.warning(f"⚠️ [STREAM PERSONALIZATION] Error formatting profile: {e}")
            
            rag_prompt = ChatPrompts.get_medical_rag_prompt(
                query=query,
                personal_info=user_context,
                medical_history=context_pieces,
                intake_form_context=intake_form_context,
                lab_report_summary=lab_report_summary,
                health_data_context=health_data_context,
                memories=memories
            )
            
            if personalization_context:
                rag_prompt += f"\n\n=== PERSONALIZED HEALTH PROFILE ===\n{personalization_context}\n\nUse this personalized profile to tailor your responses to the user's specific health goals, lifestyle, and therapeutic focus areas.\n"
            # ================= Scar STREAMING CLINICAL SAFETY =================

            system_safety_block = """
            You are Scar, a compassionate health companion.

            You MUST follow these medical communication principles:
            - Do not provide diagnoses unless explicitly requested.
            - Never present advice as a substitute for professional medical consultation.
            - Use empathetic, calm language.
            - If symptoms may indicate risk, recommend consulting a doctor.
            - For high risk, include urgency such as:
            "Please seek immediate medical attention."

            Tone:
            - Supportive
            - Non-judgmental
            - Reassuring but not dismissive
            - Clear and grounded
            """

            rag_prompt = system_safety_block + "\n\n" + rag_prompt

            # HIGH RISK ESCALATION
            if medical_risk_level == "HIGH":
                rag_prompt += """
            IMPORTANT SAFETY NOTE:
            This user's symptoms may indicate a serious medical concern.
            Strongly advise urgent medical evaluation or emergency services.
            """

            # SYMPTOM MODE
            if health_type == "SYMPTOM":
                rag_prompt += """
            When responding, ask structured follow-up questions:
            - Duration of symptoms
            - Severity
            - Triggers
            - Associated symptoms
            """

            logger.info("💬 [STREAM CHAT] Starting streaming response generation...")
            
            full_response = ""
            async for chunk in self.llm.astream([HumanMessage(content=rag_prompt)]):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    yield {"type": "response_chunk", "content": chunk.content}
                    # Small delay to allow UI thread to process
                    await asyncio.sleep(0.01)
            
            logger.info("💬 [STREAM CHAT] Generating follow-up questions...")
            if full_response and not ("sorry" in full_response.lower() or "error" in full_response.lower()):
                follow_up_prompt = ChatPrompts.get_follow_up_questions_prompt(query, full_response)
                try:
                    # Use async LLM call to avoid blocking
                    response = await self.llm.ainvoke([HumanMessage(content=follow_up_prompt)])
                    follow_up_questions = [q.strip().lstrip("- ").lstrip("* ") for q in response.content.strip().split('\n') if q.strip()]
                    yield {"type": "follow_up", "content": follow_up_questions[:4]}
                except Exception as e:
                    logger.error(f"Error generating follow-up questions: {e}")
                    yield {"type": "follow_up", "content": []}
            else:
                yield {"type": "follow_up", "content": []}
            
            # Store conversation in memory after successful completion
            if full_response:
                try:
                    messages_to_store = []
                    if conversation_history:
                        messages_to_store.extend(conversation_history)
                    # Add current interaction
                    messages_to_store.append({"role": "user", "content": query})
                    messages_to_store.append({"role": "assistant", "content": full_response})
                    
                    await self.memory_service.add_memory(user_email, messages_to_store)
                    logger.info(f"🧠 [MEMORY] Stored conversation in memory for user {user_email}")
                except Exception as e:
                    logger.warning(f"⚠️ [MEMORY] Failed to store memories: {e}")
                
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            yield {"type": "error", "content": "An error occurred during streaming."}

chat_service = None

def get_chat_service() -> ChatService:
    global chat_service
    if chat_service is None:
        chat_service = ChatService()
    return chat_service
