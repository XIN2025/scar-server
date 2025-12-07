"""
Centralized prompts for the chat service
"""
from typing import List, Dict, Any
import json
from app.schemas.ai.lab_report import LabReport
from app.schemas.backend.health_alert import HealthData

class ChatPrompts:
    """Collection of prompts used in the chat service"""
    
    @staticmethod
    def get_query_classification_prompt(query: str) -> str:
        """Get the prompt for classifying if a query needs RAG or just chat"""
        return f"""
        Classify the following query to determine if it needs information retrieval or just conversational response.
        
        Query: {query}
        
        If the query is asking for specific information, facts, or knowledge that might be in a knowledge base, respond with "rag".
        If the query is just casual conversation, greetings, or general chat, respond with "chat".
        
        Respond with only "rag" or "chat".
        """
    
    @staticmethod
    def get_medical_rag_prompt(
        query: str,
        personal_info: str,
        medical_history: List[str],
        intake_form_context: str = "",
        lab_report_summary: str = "",
        health_data_context: str = "",
        memories: List[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a prompt for a medical AI using detailed patient data.

        Args:
            query: The user's medical question.
            personal_info: A string containing the user's personal information (e.g., "Patient's name is John. Age is 30.").
            medical_history: A list of strings representing the user's relevant past medical history related to the query.
            intake_form_context: Formatted string of intake form data (age, gender, goals, conditions, etc.).
            lab_report_summary: AI-generated summary of lab report results.
            health_data_context: Formatted string of recent health metrics and hourly data.

        Returns:
            A formatted prompt string for the language model.
        """
        
        # Format medical_history list into readable string
        if medical_history:
            filtered_history = [item.strip() for item in medical_history if item and item.strip()]
            medical_history_str = "\n\n".join(filtered_history) if filtered_history else "No relevant medical history available."
        else:
            medical_history_str = "No relevant medical history available."
        
        # Build context sections
        context_sections = []
        
        if intake_form_context:
            context_sections.append(f"PATIENT INTAKE FORM:\n---\n{intake_form_context}\n---")
        
        if lab_report_summary:
            context_sections.append(f"LAB REPORTS SUMMARY:\n---\n{lab_report_summary}\n---")
        
        if health_data_context:
            context_sections.append(f"RECENT HEALTH METRICS:\n---\n{health_data_context}\n---")
        
        # Format memories from previous conversations
        if memories and len(memories) > 0:
            memory_texts = []
            for memory in memories:
                memory_content = memory.get("memory", "") or memory.get("content", "")
                if memory_content:
                    memory_texts.append(f"- {memory_content}")
            if memory_texts:
                memories_str = "\n".join(memory_texts)
                context_sections.append(f"PREVIOUS CONVERSATION MEMORIES:\n---\n{memories_str}\n---")
        
        additional_context = "\n\n".join(context_sections) if context_sections else "No additional context available."
        
        return f"""You are a clinical medical assistant providing evidence-based medical information.

CRITICAL INSTRUCTIONS:
1. Use FORMAL, PROFESSIONAL language throughout
2. NEVER use closings like "Best regards", "Wishing you well", "Hope this helps", etc.
3. MATCH YOUR RESPONSE LENGTH TO THE QUESTION COMPLEXITY:
   - Simple questions (1-2 sentences) → Brief answer (2-4 sentences max)
   - Moderate questions → Concise answer (1-2 paragraphs max)
   - Complex questions → Detailed but focused answer (3-4 paragraphs max)
4. NEVER write essays or long explanations for simple questions
5. Answer DIRECTLY and stop - no elaboration unless the question specifically asks for it
6. Do NOT add pleasantries, casual remarks, or personal sentiments
7. End with the final point - no signatures, closings, or additional text
8. When available, reference specific data from the patient's intake form, lab reports, or health metrics to provide personalized insights
9. FORMAT YOUR RESPONSE USING MARKDOWN FOR READABILITY:
   - Use headings (###) for major sections when helpful
   - Use **bold** for key metrics/values and important warnings
   - Use bullet lists (-) for recommendations or multiple points
   - Use numbered lists (1., 2.) for stepwise guidance
   - Insert blank lines between paragraphs/lists for spacing

PATIENT INFORMATION:
---
{personal_info}
---

RETRIEVED DOCUMENTS (relevant to query):
---
{medical_history_str}
---

{additional_context}

USER'S QUESTION:
{query}

Instructions:
- If this is a greeting or simple conversational message (hi, hello, thanks, okay, etc.), respond naturally, warmly, and briefly as a health-focused assistant. Be friendly but professional.
- If this is a health question, provide a response APPROPRIATE IN LENGTH for the specific question. Keep simple questions brief, provide details for complex ones. Never over-explain.
- When relevant, incorporate insights from the patient's intake form, lab reports, and recent health metrics.
- For greetings, you can relax the formal tone slightly while maintaining professionalism.
- Ensure the final output follows the markdown formatting rules above when applicable (may skip markdown for simple greetings)."""

    @staticmethod
    def get_rag_reasoning_prompt(initial_response: str, query: str, context: list) -> str:
        """Get the prompt for reasoning through and enhancing the initial RAG response"""
        return f"""
        You are an expert AI assistant. You have provided an initial response to a user's question based on retrieved context. Now, please reason through your response and enhance it with your knowledge and expertise.
        
        User Question: {query}
        
        Retrieved Context:
        {chr(10).join(context)}
        
        Your Initial Response:
        {initial_response}
        
        Now, please:
        1. Review your initial response critically
        2. Consider if there are gaps in the information from the context
        3. Enhance the response with your general knowledge and expertise
        4. Provide additional insights, explanations, or clarifications that would be helpful
        5. Ensure the response is comprehensive, well-structured, and addresses the user's question thoroughly
        6. If the context is insufficient, acknowledge this and provide what you can from your knowledge
        
        Provide a well-reasoned, enhanced response that combines the retrieved information with your expertise.
        """
    
    @staticmethod
    def get_conversational_response_prompt(query: str) -> str:
        """Get the prompt for generating conversational responses"""
        return f"""You are a clinical medical assistant.

CRITICAL INSTRUCTIONS:
1. Use FORMAL, PROFESSIONAL language only
2. MATCH response length to question complexity - simple questions get brief answers
3. NO friendly closings like "Best regards", "Wishing you well"
4. Answer directly and stop - no essays unless specifically needed
5. End with the final point - no signatures or additional text

User query: {query}

Provide an appropriately sized response - brief for simple questions, detailed only if needed."""
    
    @staticmethod
    def get_follow_up_questions_prompt(query: str, response: str) -> str:
        """Get the prompt for generating follow-up questions"""
        return f"""
        Based on the user's question and your response, generate 3-4 relevant follow-up questions that the user might want to ask next.
        
        User Question: {query}
        Your Response: {response}
        
        Generate follow-up questions that are:
        1. Relevant to the topic
        2. Natural conversation flow
        3. Helpful for the user
        4. Different from each other
        
        Return only the questions, one per line, without numbering or formatting.
        """

    @staticmethod
    def get_health_alerts_prompt(threshold_alerts: list, previous_health_alerts: list) -> str:
        """
        Generate the prompt for health alerts analysis based on threshold alerts and previous alerts.
        
        Args:
            threshold_alerts (list): List of threshold-based alerts that were generated.
            previous_health_alerts (list): List of previous alert dicts with created_at timestamps.
        
        Returns:
            str: The formatted prompt string.
        """
        from datetime import datetime, timezone, timedelta
        
        now_utc = datetime.now(timezone.utc)
        two_hours_ago = now_utc - timedelta(hours=1)
        
        prompt = f"""
           **CRITICAL RULES FOR ALERT FILTERING AND GENERATION:**
            1.  **Filter Threshold Alerts:** You have been provided with `THRESHOLD ALERTS` below. You MUST filter these alerts based on `PREVIOUS ALERTS`.
            2.  **Exclude Recent Duplicates:** If a threshold alert has the same `metric` as any alert in `PREVIOUS ALERTS` that was created within the last 2 hours (created_at >= {two_hours_ago.isoformat()}), you MUST exclude that threshold alert from your response.
            3.  **Return Only Filtered Alerts:** Only return threshold alerts that do NOT have a matching metric in recent previous alerts (within last 2 hours).
            4.  **If All Filtered Out:** If all threshold alerts are filtered out (all match recent previous alerts), set `should_generate_alert=false` and return an empty alerts array.

            ---
            **THRESHOLD ALERTS (These are the alerts you need to filter):**
            {json.dumps(threshold_alerts, indent=2, default=str)}

            ---
            **PREVIOUS ALERTS (Check created_at timestamp - exclude threshold alerts with matching metrics created within last 2 hours):**
            {json.dumps(previous_health_alerts, indent=2, default=str)}

            ---
            **RESPONSE FORMAT:**
            -   Set `should_generate_alert=true` ONLY if you have threshold alerts that don't match recent previous alerts (within last 2 hours).
            -   Return ONLY the filtered threshold alerts in the `alerts` array.
            -   Each alert entry must include:
                -   `metric`: The metric name (must match exactly from threshold alerts).
                -   `title`: The title from the threshold alert.
                -   `key_point`: The key point from the threshold alert.
                -   `message`: The message from the threshold alert.
                -   `severity`: The severity from the threshold alert ('high', 'medium', or 'low').
            """
        return prompt

    @staticmethod
    def get_lab_report_score_prompt(lab_report: LabReport) -> str:
        """Get the prompt for scoring a lab report"""
        lab_report_json = lab_report.model_dump()
        return f"""
        You are a medical AI assistant evaluating a lab report for overall health assessment.
        
        Your task is to analyze this lab report and determine if it represents overall GOOD health or NOT GOOD health.
        
        IMPORTANT EVALUATION CRITERIA:
        1. **Overall Health Assessment**: Consider the report as a whole, not individual values in isolation
        2. **Proportion of Normal Values**: If 80%+ of values are within normal range, the report is likely "Good"
        3. **Severity of Abnormalities**: 
           - Minor borderline abnormalities (slightly high/low) should not automatically make it "Not Good"
           - Only significant abnormalities or multiple concerning patterns should result in "Not Good"
        4. **Critical Markers**: Pay special attention to critical health markers (HbA1c, cholesterol, cardiac markers, liver/kidney function)
        5. **Clinical Context**: A report with mostly normal values and a few minor abnormalities is generally "Good"
        
        SCORING GUIDELINES:
        - Score as "Good" if:
          * 80%+ of values are within normal range
          * Abnormalities are minor/borderline and not clinically significant
          * Critical markers (HbA1c, cholesterol, etc.) are within acceptable ranges
          * No concerning patterns or multiple severe abnormalities
        
        - Score as "Not Good" if:
          * Less than 70% of values are within normal range
          * Critical markers show significant abnormalities (e.g., HbA1c > 6.5, high cholesterol)
          * Multiple severe abnormalities indicating health concerns
          * Clear patterns suggesting underlying health issues
        
        Return a JSON with:
        - score: Either "Good" or "Not Good"
        - reasons: List of strings explaining your assessment (focus on overall health, not individual values)
        
        Lab report data:
        {lab_report_json}
        """

    @staticmethod
    def get_health_data_score_prompt(health_data: HealthData) -> str:
        """Get the prompt for scoring health data"""
        health_data_json = health_data.model_dump()
        return f"""
        You are a medical AI. Given this health data, 
        analyze if the data is overall healthy. 
        Return a JSON with keys: score (0-100), and reasons (list of strings). 
        Health data: {health_data_json}
        """

    @staticmethod
    def get_health_relevance_classification_prompt(query: str) -> str:
        """Get the prompt for classifying if a query is health-related"""
        return f"""You are a medical AI assistant classifier. Analyze this query and determine if it should be processed.

Query: "{query}"

Classify as "health_related" ONLY if:
- The query is DIRECTLY about: medical conditions, symptoms, treatments, medications, health advice, wellness, fitness, nutrition, lab results, medical tests, diagnoses, mental health, emotional wellbeing, physical health, body functions, medical procedures.
- The query is a PURE greeting or acknowledgment with NO other content (e.g., "hi", "hello", "thanks", "okay", "got it").
- The query ONLY asks about your capabilities as a health assistant (e.g., "what can you do?", "how can you help me?", "what are you?").

Classify as "not_health_related" if:
- The query asks INFORMATION-SEEKING questions about non-health topics, even if it starts with a greeting (e.g., "hi how can i become president", "tell me about games", "what is the capital of France", "explain quantum physics").
- The query is about: general knowledge facts, history, science (non-medical), technology, programming, business, entertainment, sports facts, news events, politics (elections, government, presidency), economics, social issues, careers (non-health), weather, travel, cooking recipes (non-health).

CRITICAL: If a query combines a greeting with a substantive question, classify based on the ACTUAL QUESTION content, not the greeting.
Examples:
- "Hi" → health_related (pure greeting)
- "Hi how can i become president" → not_health_related (asks about politics/careers)
- "Hello, what is the capital of France?" → not_health_related (asks about geography)
- "Hi, I have a headache" → health_related (asks about health symptom)

Respond with ONLY one word: "health_related" or "not_health_related"
"""

    @staticmethod
    def get_non_health_response_prompt() -> str:
        """Get the prompt for responding to non-health related questions"""
        return """
        You are a health-focused AI assistant. The user has asked a question that is not related to health, medicine, or wellness.
        
        Provide a polite, minimal response that:
        1. Acknowledges that you're designed for health guidance
        2. Politely redirects them to appropriate platforms for their question
        3. Keeps the response brief and professional
        
        Suggested platforms to mention (use only 2-3 most relevant):
        - For general knowledge: Wikipedia, Google
        - For technology: Stack Overflow, GitHub
        - For news: BBC, CNN, Reuters
        - For entertainment: IMDB, Rotten Tomatoes
        - For education: Khan Academy, Coursera
        
        Keep the response under 50 words and be helpful but firm about your health focus.
        
        Example responses:
        - "I'm designed to help with health guidance. For weather information, try Weather.com or your local weather app."
        - "I focus on health and medical questions. For technology help, try Stack Overflow or GitHub."
        - "I'm here for health support. For general knowledge, Wikipedia or Google would be better resources."
        """

def get_prompts():
    return ChatPrompts()
