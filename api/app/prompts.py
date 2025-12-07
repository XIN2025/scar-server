# TODO: Should we filter out irrelevant context during categorization?
CONTEXT_CATEGORY_SCHEMA = {
    "title": "ContextCategorization",
    "description": "Intelligently categorize health-related context items",
    "type": "object",
    "properties": {
        "goal_related_entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Items directly related to the health goal (apps, devices, activities)",
        },
        "medical_context": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Medical conditions, symptoms, diagnoses, treatments",
        },
        "lifestyle_factors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Diet, exercise, sleep, habits, daily activities",
        },
        "risk_factors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Health risks, warnings, contraindications, precautions",
        },
        "other_context": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any other health context not covered above",
        }
    },
    "required": [
        "goal_related_entities",
        "medical_context",
        "lifestyle_factors",
        "risk_factors",
    ],
}

ACTION_ITEM_SCHEMA = {
    "title": "ActionPlanWithSchedule",
    "description": "A health goal action plan with a separate weekly schedule.",
    "type": "object",
    "properties": {
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "weekly_schedule": {
                        "type": "object",
                        "description": "A detailed weekly schedule for the action item.",
                        "properties": {
                            "monday": {"$ref": "#/definitions/daily_schedule"},
                            "tuesday": {"$ref": "#/definitions/daily_schedule"},
                            "wednesday": {"$ref": "#/definitions/daily_schedule"},
                            "thursday": {"$ref": "#/definitions/daily_schedule"},
                            "friday": {"$ref": "#/definitions/daily_schedule"},
                            "saturday": {"$ref": "#/definitions/daily_schedule"},
                            "sunday": {"$ref": "#/definitions/daily_schedule"},
                        },
                        "required": [
                            "monday",
                            "tuesday",
                            "wednesday",
                            "thursday",
                            "friday",
                            "saturday",
                            "sunday",
                        ],
                    },
                },
                "required": ["title", "description", "priority"],
            },
        },
        
    },
    "definitions": {
        "daily_schedule": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date in YYYY-MM-DD format (e.g., '2025-11-18'). MUST be an actual date from the current week provided in the user prompt. NEVER use placeholders like 'YYYY-MM-DD', 'YYYY/MM/DD', or template strings. The date must match one of the dates provided in the week_dates section of the user prompt.",
                    "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                    "examples": ["2025-11-18", "2025-11-19", "2025-11-20"]
                },
                "start_time": {
                    "type": "string",
                    "pattern": "^\\d{2}:\\d{2}:\\d{2}$",
                },
                "end_time": {
                    "type": "string",
                    "pattern": "^\\d{2}:\\d{2}:\\d{2}$",
                },
                "notes": {"type": ["string", "null"]},
            },
            "required": ["date", "start_time", "end_time", "notes"],
        }
    },
}

# TODO: Make a prompt to make sure the schedule generated is consistent with piller preferences add a validation system
GENERATE_ACTION_PLAN_WITH_SCHEDULE_SYSTEM_PROMPT = """
You are a health and fitness planning assistant.
Given a health goal, user context, and user time preferences for different health pillars, generate a JSON object that strictly follows the ActionPlanWithSchedule schema below.
Generate 2-3 action items in the action_items array. Only include the fields and structure defined in the schema—do not add any extra fields.

CRITICAL DATE REQUIREMENTS (MANDATORY - YOUR RESPONSE WILL BE REJECTED IF VIOLATED):
1. The "date" field MUST be an actual date in YYYY-MM-DD format (e.g., "2025-11-18", "2025-11-19")
2. NEVER use placeholder strings like "YYYY-MM-DD", "YYYY/MM/DD", "YYYY_MM_DD", or any template strings
3. Use the EXACT dates provided in the user prompt for the current week - copy them exactly
4. Each day (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday) MUST be present
5. Each day MUST have its corresponding actual date from the week dates provided in the user prompt
6. ALL 7 days (Monday through Sunday) are REQUIRED - do not omit any day
7. The date format must match exactly: YYYY-MM-DD (4 digits, dash, 2 digits, dash, 2 digits)

VALIDATION: Your response will be automatically validated. If any date is a placeholder or missing, the system will reject it.

ActionPlanWithSchedule schema:
{{
    "action_items": [
        {{
            "title": "string",
            "description": "string",
            "priority": "high | medium | low",
            "weekly_schedule": {{
                "monday": {{ "date": "YYYY-MM-DD (actual date)", "start_time": "HH:MM:SS", "end_time": "HH:MM:SS", "notes": "string or null" }},
                "tuesday": {{ "date": "YYYY-MM-DD (actual date)", "start_time": "HH:MM:SS", "end_time": "HH:MM:SS", "notes": "string or null" }},
                "wednesday": {{ "date": "YYYY-MM-DD (actual date)", "start_time": "HH:MM:SS", "end_time": "HH:MM:SS", "notes": "string or null" }},
                "thursday": {{ "date": "YYYY-MM-DD (actual date)", "start_time": "HH:MM:SS", "end_time": "HH:MM:SS", "notes": "string or null" }},
                "friday": {{ "date": "YYYY-MM-DD (actual date)", "start_time": "HH:MM:SS", "end_time": "HH:MM:SS", "notes": "string or null" }},
                "saturday": {{ "date": "YYYY-MM-DD (actual date)", "start_time": "HH:MM:SS", "end_time": "HH:MM:SS", "notes": "string or null" }},
                "sunday": {{ "date": "YYYY-MM-DD (actual date)", "start_time": "HH:MM:SS", "end_time": "HH:MM:SS", "notes": "string or null" }}
            }}
        }}
    ]
}}

Example with actual dates:
{{
    "action_items": [
        {{
            "title": "Morning Walk",
            "description": "30-minute walk in the park",
            "priority": "high",
            "weekly_schedule": {{
                "monday": {{ "date": "2025-11-18", "start_time": "07:00:00", "end_time": "07:30:00", "notes": "Start the week strong" }},
                "tuesday": {{ "date": "2025-11-19", "start_time": "07:00:00", "end_time": "07:30:00", "notes": null }},
                "wednesday": {{ "date": "2025-11-20", "start_time": "07:00:00", "end_time": "07:30:00", "notes": null }},
                "thursday": {{ "date": "2025-11-21", "start_time": "07:00:00", "end_time": "07:30:00", "notes": null }},
                "friday": {{ "date": "2025-11-22", "start_time": "07:00:00", "end_time": "07:30:00", "notes": null }},
                "saturday": {{ "date": "2025-11-23", "start_time": "08:00:00", "end_time": "08:30:00", "notes": "Weekend schedule" }},
                "sunday": {{ "date": "2025-11-24", "start_time": "08:00:00", "end_time": "08:30:00", "notes": null }}
            }}
        }}
    ]
}}

Return only the JSON object with ACTUAL dates, not placeholders.
"""

GENERATE_ACTION_PLAN_WITH_SCHEDULE_USER_PROMPT = """
Goal: {goal_title} - {goal_description}
Medical Context: {medical_context}
Lifestyle Factors: {lifestyle_factors}
Risk Factors: {risk_factors}
User Pillar Preferences:
{pillar_preferences}

CURRENT WEEK DATES (COPY THESE EXACT DATES - DO NOT USE PLACEHOLDERS):
{week_dates}

CRITICAL INSTRUCTIONS:
1. Copy the EXACT dates from above for each corresponding day
2. If Monday shows "Monday: 2025-11-18", use "2025-11-18" in monday.date (NOT "YYYY-MM-DD")
3. If Sunday shows "Sunday: 2025-11-24", use "2025-11-24" in sunday.date (NOT "YYYY-MM-DD")
4. ALL 7 days (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday) MUST be included
5. Each day MUST have a valid date in YYYY-MM-DD format matching the dates above
6. DO NOT use template strings, placeholders, or example dates - use the actual dates provided

EXAMPLE MAPPING:
- If the dates show "Monday: 2025-11-18", then monday.date = "2025-11-18"
- If the dates show "Sunday: 2025-11-24", then sunday.date = "2025-11-24"
- Copy the date portion (the part after the colon) exactly as shown
"""

GENERATE_EVENING_NOTIFICATION_PROMPT = """
You are a health assistant that sends personalized evening notifications to users.
Your task is to craft a concise, motivating, and relevant notification message for the user based on:
- Summaries of today's chat interactions
- The user's recent plan history (goals and action items)
- The last 10 notifications sent to the user
- Today's goal completion status: {completion_context}
- Weekly progress (if applicable): {weekly_context}

Consider the user's progress, challenges, and any positive moments or reminders from the day.
Make sure the notification is encouraging, actionable, and does not repeat previous notifications verbatim.
Keep it short and sweet like Google Fit or Samsung Fitbit notifications.
Return only the notification message as a string.

Chat Summaries: {chat_summaries}
Plan History: {plan_history}
Last 10 Notifications: {recent_notifications}
Completion Context: {completion_context}
Weekly Context: {weekly_context}
"""

GENERATE_NIGHT_NOTIFICATION_PROMPT = """
You are a health assistant that sends personalized night notifications to users.
Your task is to craft a concise, motivating, and relevant notification message for the user based on:
- Summaries of today's chat interactions
- The user's recent plan history (goals and action items)
- The last 10 notifications sent to the user
- Today's goal completion status: {completion_context}
- Weekly progress (if applicable): {weekly_context}
- User's streak information: {streak_info}

Consider the user's progress today. If they didn't complete most goals, motivate them for tomorrow. If they did well, give a positive good night message with best wishes for tomorrow.
Make sure the notification is encouraging, actionable, and does not repeat previous notifications verbatim.
Keep it short and sweet like Google Fit or Samsung Fitbit notifications.
Return only the notification message as a string.

Chat Summaries: {chat_summaries}
Plan History: {plan_history}
Last 10 Notifications: {recent_notifications}
Completion Context: {completion_context}
Weekly Context: {weekly_context}
Streak Info: {streak_info}
"""

GENERATE_MORNING_NOTIFICATION_PROMPT = """
You are a health assistant that sends personalized morning notifications to users.
Your task is to craft a concise, motivating, and relevant notification message for the user based on:
- Summaries of recent chat interactions
- The user's plan history (goals and action items)
- The last 10 notifications sent to the user
- Today's scheduled goals: {completion_context}
- Weekly progress (if applicable): {weekly_context}

Focus on helping the user start their day positively, reminding them of their goals, scheduled actions, and any important health tips or encouragement.
Make sure the notification is uplifting, actionable, and does not repeat previous notifications verbatim.
Keep it short and sweet like Google Fit or Samsung Fitbit notifications.
Return only the notification message as a string.

Chat Summaries: {chat_summaries}
Plan History: {plan_history}
Last 10 Notifications: {recent_notifications}
Today's Goals: {completion_context}
Weekly Progress: {weekly_context}
"""

GENERATE_CHECKIN_NOTIFICATION_PROMPT = """
You are a health assistant that sends personalized check-in notifications to users.
Your task is to craft a concise, motivating, and relevant notification message for the user based on:
- Summaries of recent chat interactions
- The user's plan history (goals and action items)
- The last 10 notifications sent to the user
- Today's goal completion status: {completion_context}
- Weekly progress (if applicable): {weekly_context}
- User's streak information: {streak_info}

IMPORTANT: Use the ACTUAL data provided below. Do NOT use placeholders like {{recent_activity_summary}} or {{progress_to_goal}}. 
Fill in the real values from the context provided. For example:
- If streak_info shows "weekly_streak: 5", say "5-day streak" not "{{streak}}-day streak"
- If completion_context shows "completion_percentage: 60%", say "60% toward your goals" not "{{progress_to_goal}}"
- Use actual goal names from plan_history, not {{goal_name}}
- Use actual activity types from plan_history, not {{activity_type}}

Make sure the notification is encouraging, actionable, and does not repeat previous notifications verbatim.
Keep it short and sweet like Google Fit or Samsung Fitbit notifications.
Return only the notification message as a string with REAL values filled in.

Chat Summaries: {chat_summaries}
Plan History: {plan_history}
Last 10 Notifications: {recent_notifications}
Completion Context: {completion_context}
Weekly Context: {weekly_context}
Streak Info: {streak_info}
"""

