# TODO: Can we use encrypt_bulk_documents or decrypt_bulk_documents instead of iterating an array of documents

from datetime import datetime, timedelta, time, date, timezone
from pprint import pprint
from typing import List, Optional, Dict, Any, Tuple
from bson import ObjectId
import asyncio
from ...schemas.ai.goals import GoalUpdate, Goal, WeeklyReflectionCreate, WeeklyReflection, GoalStats, GoalWithActionItems, ActionItem, ActionItemCreate, ActionPriority, WeeklyActionSchedule, DailySchedule, StreakScore, GoalCreate
from ...schemas.backend.preferences import PillarTimePreferences
from ...schemas.backend.action_completions import (
    ActionItemCompletion, ActionItemCompletionCreate, ActionItemCompletionUpdate,
    DailyCompletionStats, WeeklyCompletionStats
)
from app.services.ai_services.mongodb_vectorstore import get_vector_store
from ..backend_services.db import get_db
from app.prompts import (
    CONTEXT_CATEGORY_SCHEMA,
    ACTION_ITEM_SCHEMA,
    GENERATE_ACTION_PLAN_WITH_SCHEDULE_SYSTEM_PROMPT,
    GENERATE_ACTION_PLAN_WITH_SCHEDULE_USER_PROMPT,
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import OPENAI_API_KEY, LLM_MODEL
from calendar import monthrange
from app.services.backend_services.encryption_service import get_encryption_service
import logging

logger = logging.getLogger(__name__)

class GoalsService:
    def __init__(self):
        self.db = get_db()
        self.goals_collection = self.db["goals"]
        self.action_items_collection = self.db["action_items"]
        self.reflections_collection = self.db["weekly_reflections"]
        self.encryption_service = get_encryption_service()
        self.vector_store = get_vector_store()

    async def _invoke_structured_llm(
        self, schema: dict, system_prompt: str, user_prompt: str, input_vars: dict
    ) -> dict:
        llm = ChatOpenAI(
            model=LLM_MODEL, openai_api_key=OPENAI_API_KEY
        ).with_structured_output(schema)
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt)]
        )
        chain = prompt | llm
        return await chain.ainvoke(input_vars)

    # TODO: Complete the schema for daily completion and daily completion response
    # TODO: Add user_email in action_items schema because it is needed here to filter action items by user
    async def get_current_daily_completion(self, user_email: str, month: int, year: int) -> Dict[str, int]:
        start_date = datetime(year, month, 1, tzinfo=timezone.utc)
        last_day = monthrange(year, month)[1]
        end_date = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
        action_items = await self.action_items_collection.find({
            "user_email": user_email
        }).to_list(None)
        action_items = self.encryption_service.decrypt_documents_bulk(action_items, ActionItem)
        daily_counts = {}
        for item in action_items:
            weekly_schedule = item.get("weekly_schedule", {})
            if not weekly_schedule:
                continue
            for daily_schedule in weekly_schedule.values():
                if not daily_schedule:
                    continue
                completed = daily_schedule.get("complete", False)
                date_value = daily_schedule.get("date")
                
                # Safely parse date - handle both datetime objects and strings
                action_item_date = None
                if isinstance(date_value, datetime):
                    action_item_date = date_value.replace(tzinfo=timezone.utc) if date_value.tzinfo is None else date_value
                elif isinstance(date_value, str):
                    # Skip placeholder/template strings like 'YYYY-MM-DD'
                    if date_value.upper() in ['YYYY-MM-DD', 'YYYY/MM/DD', 'YYYY_MM_DD']:
                        continue
                    # Validate and parse date string
                    try:
                        action_item_date = datetime.strptime(date_value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    except ValueError as e:
                        logger.warning(f"Invalid date format in action item for user {user_email}: {date_value}, error: {str(e)}")
                        continue
                else:
                    continue  # Skip if date is invalid type
                
                if action_item_date is None:
                    continue
                    
                if completed and start_date <= action_item_date <= end_date:
                    day_str = action_item_date.strftime("%Y-%m-%d")
                    daily_counts[day_str] = daily_counts.get(day_str, 0) + 1
        return daily_counts

    # TODO: This function can actually replace get_current_daily_completion because it is more generic
    async def get_all_daily_completions(self, user_email: str) -> Dict[str, int]:
        action_items = await self.action_items_collection.find({
            "user_email": user_email
        }).to_list(None)
        action_items = self.encryption_service.decrypt_documents_bulk(action_items, ActionItem)

        daily_counts = {}
        for item in action_items:
            weekly_schedule = item.get("weekly_schedule", {})
            if not weekly_schedule:
                continue
            for daily_schedule in weekly_schedule.values():
                if not daily_schedule:
                    continue
                completed = daily_schedule.get("complete", False)
                date_value = daily_schedule.get("date")
                
                # Safely parse date - handle both datetime objects and strings
                action_item_date = None
                if isinstance(date_value, datetime):
                    action_item_date = date_value.replace(tzinfo=timezone.utc) if date_value.tzinfo is None else date_value
                elif isinstance(date_value, str):
                    # Skip placeholder/template strings like 'YYYY-MM-DD'
                    if date_value.upper() in ['YYYY-MM-DD', 'YYYY/MM/DD', 'YYYY_MM_DD']:
                        continue
                    # Validate and parse date string
                    try:
                        action_item_date = datetime.strptime(date_value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    except ValueError as e:
                        logger.warning(f"Invalid date format in action item for user {user_email}: {date_value}, error: {str(e)}")
                        continue
                else:
                    continue  # Skip if date is invalid type
                
                if action_item_date is None:
                    continue
                    
                if completed:
                    day_str = action_item_date.strftime("%Y-%m-%d")
                    daily_counts[day_str] = daily_counts.get(day_str, 0) + 1
        return daily_counts

    async def create_goal(self, goal_data: GoalCreate) -> Goal:
        goal_data = self.encryption_service.encrypt_document(goal_data, GoalCreate)
        goal_dict = goal_data.model_dump()
        goal_dict["_id"] = ObjectId()
        goal_dict["created_at"] = datetime.now(timezone.utc)
        result = await self.goals_collection.insert_one(goal_dict)
        goal_dict["id"] = str(result.inserted_id)
        del goal_dict["_id"]  
        goal_dict = self.encryption_service.decrypt_document(goal_dict, Goal)
        return Goal(**goal_dict)

    async def get_user_goals(self, user_email: str) -> List[Dict[str, Any]]:
        # Fetch all goals for the user
        goals = await self.goals_collection.find({
            "user_email": user_email
        }).to_list(None)
        goals = self.encryption_service.decrypt_documents_bulk(goals, Goal)

        # Extract goal IDs for batch query
        goal_ids = [str(goal["_id"]) for goal in goals]
        
        # Batch fetch all action items for all goals in a single query (fixes N+1 problem)
        all_action_items = []
        if goal_ids:
            all_action_items = await self.action_items_collection.find({
                "goal_id": {"$in": goal_ids}
            }).to_list(None)
            all_action_items = self.encryption_service.decrypt_documents_bulk(all_action_items, ActionItem)

        action_items_by_goal = {}
        for action_item in all_action_items:
            if isinstance(action_item, ActionItem):
                action_item_dict = action_item.model_dump()
                goal_id = action_item_dict.get("goal_id")
                if "_id" in action_item_dict:
                    action_item_dict["id"] = str(action_item_dict["_id"])
                    del action_item_dict["_id"]
                action_item_obj = ActionItem(**action_item_dict)
            else:
                goal_id = action_item.get("goal_id")
                action_item["id"] = str(action_item["_id"])
                del action_item["_id"]
                action_item_obj = ActionItem(**action_item)
            
            if goal_id:
                if goal_id not in action_items_by_goal:
                    action_items_by_goal[goal_id] = []
                action_items_by_goal[goal_id].append(action_item_obj)

        # Build response with action items grouped by goal
        goals_with_action_items = []
        for goal in goals:
            goal["id"] = str(goal["_id"])
            del goal["_id"]
            # Items are already ActionItem instances
            goal["action_items"] = action_items_by_goal.get(goal["id"], [])
            goals_with_action_items.append(GoalWithActionItems(**goal))
        
        goals_with_action_items.sort(key=lambda g: g.created_at, reverse=True)
            
        return goals_with_action_items

    async def get_goal_by_id(self, goal_id: str, user_email: str) -> Optional[Goal]:
        goal = await self.goals_collection.find_one({"_id": ObjectId(goal_id), "user_email": user_email})
        goal = self.encryption_service.decrypt_document(goal, Goal)
        if not goal:
            return None
        goal["id"] = str(goal["_id"])
        del goal["_id"]
        return Goal(**goal)

    async def delete_goal(self, goal_id: str, user_email: str) -> bool:
        try:
            obj_id = ObjectId(goal_id)
        except Exception:
            return False
        goals_deleted = await self.goals_collection.delete_one({"_id": obj_id, "user_email": user_email})
        await self.action_items_collection.delete_many({"goal_id": goal_id, "user_email": user_email})
        return goals_deleted.deleted_count > 0

    async def save_weekly_reflection(self, reflection_create: WeeklyReflectionCreate) -> WeeklyReflection:
        reflection_create = self.encryption_service.encrypt_document(reflection_create, WeeklyReflectionCreate)
        reflection_dict = reflection_create.model_dump()
        reflection_dict["created_at"] = datetime.now(timezone.utc)
        result = await self.reflections_collection.insert_one(reflection_dict)
        return WeeklyReflection(**reflection_dict, id=str(result.inserted_id))

    async def get_goal_stats(self, user_email: str) -> GoalStats:
        goals = await self.goals_collection.find({
            "user_email": user_email,
        }).to_list(None)
        goals = [self.encryption_service.decrypt_document(item, Goal) for item in goals]
        if not goals:
            return GoalStats(
                total_goals=0,
                completed_goals=0,
                completion_rate=0.0,
                average_rating=None,
                weekly_streak=0,
                total_weekly_streak_count=0
            )
        total_goals = len(goals)
        completed_goals = await self._calculate_completed_goals(user_email)
        completion_rate = await self._calculate_monthly_completion_rate(user_email)
        average_rating = await self._calculate_monthly_average_rating(user_email)
        weekly_streak = await self._calculate_current_weekly_streak(user_email)
        total_weekly_streak_count = await self._calculate_total_weekly_streak_count(user_email)
        # weekly_streak = 0
        return GoalStats(
            total_goals=total_goals,
            completed_goals=completed_goals,
            completion_rate=completion_rate,
            average_rating=average_rating,
            weekly_streak=weekly_streak,
            total_weekly_streak_count=total_weekly_streak_count
        )

    # TODO: Work on weekly streak logic because get_current_daily_completion is working on monthly basis so it might break in the end of the month
    # TODO: You can another stat for maximum stream ever or maximum monthly streak
    async def _calculate_current_weekly_streak(self, user_email: str) -> int:
        daily_completions = await self.get_current_daily_completion(user_email, datetime.now(timezone.utc).month, datetime.now(timezone.utc).year)
        if not daily_completions:
            return 0
        # Safely parse dates from keys (should be valid from strftime, but be defensive)
        dates = []
        for date_str in daily_completions.keys():
            try:
                dates.append(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc))
            except ValueError:
                logger.warning(f"Invalid date string in daily_completions keys for user {user_email}: {date_str}")
                continue
        if not dates:
            return 0
        dates = sorted(dates, reverse=True)
        streak = 0
        current_date = dates[0]
        while True:
            date_str = current_date.strftime("%Y-%m-%d")
            if daily_completions.get(date_str, 0) > 0:
                streak += 1
                current_date -= timedelta(days=1)
            else:
                break
        return streak
    
    async def _calculate_total_weekly_streak_count(self, user_email: str) -> int:
        daily_completions = await self.get_all_daily_completions(user_email)
        if not daily_completions:
            return 0
        # Safely parse dates from keys (should be valid from strftime, but be defensive)
        dates = []
        for date_str in daily_completions.keys():
            try:
                dates.append(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc))
            except ValueError:
                logger.warning(f"Invalid date string in daily_completions keys for user {user_email}: {date_str}")
                continue
        if not dates:
            return 0
        dates = sorted(dates, reverse=True)
        current_date = dates[0]
        end_date = min(dates)
        total_weeks = 0
        while current_date >= end_date:
            week_dates = {current_date - timedelta(days=i) for i in range(7)}
            is_week_complete = any(date in week_dates and daily_completions.get(date.strftime("%Y-%m-%d"), 0) > 0 for date in dates)
            if is_week_complete:
                total_weeks += 1
            current_date -= timedelta(weeks=1)
        return total_weeks

    # TODO: Logic for missing streak is not integrated yet, _calculate_current_weekly_streak needs to be updated to return missed days too
    # TODO: _calculate_current_weekly_streak is returning the latest daily streak and where we are considering it week wise
    async def get_streak_score(self, user_email: str) -> StreakScore:
        weekly_streak = await self._calculate_current_weekly_streak(user_email)
        score = 0.0
        if weekly_streak == 0:
            score = 0.0
        elif weekly_streak == 1:
            score = 10.0
        elif weekly_streak == 2:
            score = 11.0
        elif weekly_streak == 3:
            score = 12.0
        elif 4 <= weekly_streak <= 25:
            score = 12.0 + 0.5 * (weekly_streak - 3)
        else:
            score = 25.0

        score = min(score, 25.0)

        return StreakScore(week=weekly_streak, score=score)
    
    async def _calculate_monthly_average_rating(self, user_email: str) -> Optional[float]:
        start_date = datetime.now(timezone.utc) - timedelta(weeks=4)
        reflections = await self.reflections_collection.find({
            "user_email": user_email,
            "created_at": {"$gte": start_date}
        }).to_list(None)
        if not reflections:
            return None
        ratings = [r.get("rating", 0) for r in reflections if r.get("rating")]
        if not ratings:
            return None
        return sum(ratings) / len(ratings)
    
    async def _calculate_monthly_completion_rate(self, user_email: str) -> Optional[float]:
        start_date = datetime.now(timezone.utc) - timedelta(weeks=4)
        action_items = await self.action_items_collection.find({
            "user_email": user_email,
        }).to_list(None)
        if not action_items:
            return None
        total_action_items = 0
        completed_action_items = 0
        for action_item in action_items:
            weekly_schedule = action_item.get("weekly_schedule", {})
            if not weekly_schedule:
                continue
            for daily_schedule in weekly_schedule.values():
                if not daily_schedule:
                    continue
                date_value = daily_schedule.get("date")
                
                # Safely parse date - handle both datetime objects and strings
                action_item_date = None
                if isinstance(date_value, datetime):
                    action_item_date = date_value.replace(tzinfo=timezone.utc) if date_value.tzinfo is None else date_value
                elif isinstance(date_value, str):
                    # Skip placeholder/template strings like 'YYYY-MM-DD'
                    if date_value.upper() in ['YYYY-MM-DD', 'YYYY/MM/DD', 'YYYY_MM_DD']:
                        continue
                    # Validate and parse date string
                    try:
                        action_item_date = datetime.strptime(date_value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    except ValueError as e:
                        logger.warning(f"Invalid date format in action item for user {user_email}: {date_value}, error: {str(e)}")
                        continue
                else:
                    continue  # Skip if date is invalid type
                
                if action_item_date is None:
                    continue
                    
                if action_item_date < start_date:
                    continue
                total_action_items += 1
                if daily_schedule.get("completed"):
                    completed_action_items += 1
        if total_action_items == 0:
            return None
        return (completed_action_items / total_action_items) * 100

    async def _calculate_completed_goals(self, user_email: str) -> Optional[int]:
        goals = await self.goals_collection.find({
            "user_email": user_email,
        }).to_list(None)
        if not goals:
            return None
        completed_goals = 0
        for goal in goals:
            action_items = await self.action_items_collection.find({
                "goal_id": str(goal["_id"]),
                "user_email": user_email
            }).to_list(None)
            if not action_items:
                continue
            goal_completed = True
            for action_item in action_items:
                weekly_schedule = action_item.get("weekly_schedule", {})
                if not weekly_schedule:
                    continue
                for daily_schedule in weekly_schedule.values():
                    if not daily_schedule or not daily_schedule.get("completed"):
                        goal_completed = False
                        break
            if goal_completed:
                completed_goals += 1
        return completed_goals

    def _format_pillar_preferences(self, prefs: Optional[List[PillarTimePreferences]]) -> str:
        """Format pillar preferences into a readable string."""
        if not prefs:
            return "None specified"
        lines = []
        for pref in prefs:
            for pillar, time_pref in pref.preferences.items():
                days = ", ".join(str(d) for d in time_pref.days_of_week)
                lines.append(
                    f"{pillar}: Preferred time {time_pref.preferred_time}, "
                    f"Duration {time_pref.duration_minutes} min, Days {days}"
                )
        return "\n".join(lines)
    
    def _get_current_week_dates(self) -> str:
        """Get the current week's dates (Monday to Sunday) in YYYY-MM-DD format."""
        now = datetime.now(timezone.utc)
        # Get Monday of current week
        days_since_monday = now.weekday()  # 0 = Monday, 6 = Sunday
        monday = now - timedelta(days=days_since_monday)
        
        week_dates = []
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for i in range(7):
            day_date = monday + timedelta(days=i)
            date_str = day_date.strftime("%Y-%m-%d")
            week_dates.append(f"  {weekday_names[i]}: {date_str}")
        
        return "\n".join(week_dates)
    
    def _get_current_week_dates_dict(self) -> dict:
        """Get the current week's dates as a dictionary mapping day names to date strings."""
        now = datetime.now(timezone.utc)
        # Get Monday of current week
        days_since_monday = now.weekday()  # 0 = Monday, 6 = Sunday
        monday = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        weekday_names_lower = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        week_dates_dict = {}
        
        for i in range(7):
            day_date = monday + timedelta(days=i)
            date_str = day_date.strftime("%Y-%m-%d")
            week_dates_dict[weekday_names_lower[i]] = date_str
        
        return week_dates_dict

    def _validate_and_fix_llm_response(self, action_item_with_schedule: dict, week_dates_str: str) -> dict:
        """
        Validate and fix LLM response to ensure all dates are valid and all days are present.
        This is a critical guardrail to prevent placeholder dates and missing days.
        """
        week_dates_fallback = self._get_current_week_dates_dict()
        
        # Parse week dates from the string to create a mapping
        week_dates_map = {}
        for line in week_dates_str.strip().split("\n"):
            if ":" in line:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    day_name = parts[0].strip().lower()
                    date_str = parts[1].strip()
                    week_dates_map[day_name] = date_str
        
        # If parsing failed, use fallback
        if not week_dates_map:
            week_dates_map = week_dates_fallback
        
        required_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        issues_found = []
        fixes_applied = []
        
        for action_item in action_item_with_schedule.get("action_items", []):
            weekly_schedule = action_item.get("weekly_schedule", {})
            
            # Check for missing days
            missing_days = [day for day in required_days if day not in weekly_schedule]
            if missing_days:
                issues_found.append(f"Missing days: {missing_days}")
                # Create empty schedules for missing days
                for day in missing_days:
                    fallback_date = week_dates_map.get(day, week_dates_fallback.get(day))
                    if fallback_date:
                        weekly_schedule[day] = {
                            "date": fallback_date,
                            "start_time": "09:00:00",  # Default time
                            "end_time": "10:00:00",
                            "notes": None
                        }
                        fixes_applied.append(f"Added missing {day} with date {fallback_date}")
            
            # Validate and fix each day's date
            for day in required_days:
                if day not in weekly_schedule:
                    continue
                    
                day_schedule = weekly_schedule[day]
                date_str = day_schedule.get("date", "")
                
                # Check for placeholder dates
                if isinstance(date_str, str) and date_str.upper() in ['YYYY-MM-DD', 'YYYY/MM/DD', 'YYYY_MM_DD', 'YYYY-MM-DD (ACTUAL DATE)']:
                    issues_found.append(f"{day}: placeholder date '{date_str}'")
                    fallback_date = week_dates_map.get(day, week_dates_fallback.get(day))
                    if fallback_date:
                        day_schedule["date"] = fallback_date
                        fixes_applied.append(f"{day}: replaced placeholder with {fallback_date}")
                    else:
                        logger.error(f"No fallback date available for {day}")
                
                # Validate date format
                elif isinstance(date_str, str):
                    try:
                        # Try to parse the date
                        datetime.strptime(date_str, "%Y-%m-%d")
                        # Check if it matches expected week date
                        expected_date = week_dates_map.get(day, week_dates_fallback.get(day))
                        if date_str != expected_date:
                            # Date is valid but doesn't match current week - log warning but don't change
                            logger.warning(f"{day}: date {date_str} doesn't match current week date {expected_date}, but is valid")
                    except ValueError:
                        # Invalid date format
                        issues_found.append(f"{day}: invalid date format '{date_str}'")
                        fallback_date = week_dates_map.get(day, week_dates_fallback.get(day))
                        if fallback_date:
                            day_schedule["date"] = fallback_date
                            fixes_applied.append(f"{day}: replaced invalid date with {fallback_date}")
                        else:
                            logger.error(f"No fallback date available for {day}")
                else:
                    # Not a string or None
                    issues_found.append(f"{day}: date is not a string: {type(date_str)}")
                    fallback_date = week_dates_map.get(day, week_dates_fallback.get(day))
                    if fallback_date:
                        day_schedule["date"] = fallback_date
                        fixes_applied.append(f"{day}: replaced invalid date type with {fallback_date}")
        
        if issues_found:
            logger.warning(f"Validation found {len(issues_found)} issues in LLM response: {issues_found}")
        
        if fixes_applied:
            logger.info(f"Applied {len(fixes_applied)} fixes to LLM response: {fixes_applied}")
        
        return action_item_with_schedule
    
    def _normalize_schedule_times(self, action_item_with_schedule: dict) -> dict:
        """Normalize schedule times, handling None values."""
        for action_item in action_item_with_schedule.get("action_items", []):
            weekly_schedule = action_item.get("weekly_schedule", {})
            for day, schedule in weekly_schedule.items():
                if schedule.get("start_time") is None:
                    schedule["start_time"] = ""
                if schedule.get("end_time") is None:
                    schedule["end_time"] = ""
        return action_item_with_schedule

    async def _create_and_save_action_items(
        self, 
        action_item_with_schedule: dict, 
        goal_id: str, 
        user_email: str
    ) -> List[ActionItem]:
        """Create ActionItemCreate objects, encrypt, and save to database in batch."""
        action_items_to_create = []
        
        # Get current week dates as fallback
        week_dates_fallback = self._get_current_week_dates_dict()
        
        for action_item_data in action_item_with_schedule.get("action_items", []):
            try:
                # Parse weekly schedule
                weekly_schedule_dict = {}
                for day, day_schedule in action_item_data.get("weekly_schedule", {}).items():
                    try:
                        date_str = day_schedule.get("date")
                        schedule_date = None
                        
                        # Handle placeholder/template strings or invalid dates
                        if isinstance(date_str, str) and date_str.upper() in ['YYYY-MM-DD', 'YYYY/MM/DD', 'YYYY_MM_DD']:
                            logger.warning(f"Placeholder date string found for day {day}: {date_str}. Using fallback date from current week.")
                            # Use fallback date from current week
                            fallback_date = week_dates_fallback.get(day.lower())
                            if fallback_date:
                                date_str = fallback_date
                            else:
                                logger.error(f"No fallback date found for {day}, skipping")
                                continue
                        
                        # Try to parse the date
                        try:
                            schedule_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        except (ValueError, TypeError) as e:
                            # If parsing fails, use fallback date
                            logger.warning(f"Failed to parse date '{date_str}' for day {day}: {str(e)}. Using fallback date.")
                            fallback_date = week_dates_fallback.get(day.lower())
                            if fallback_date:
                                date_str = fallback_date
                                schedule_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                            else:
                                logger.error(f"No fallback date found for {day}, skipping")
                                continue
                        
                        # Parse start_time and end_time
                        start_time = None
                        if day_schedule.get("start_time") and day_schedule["start_time"]:
                            try:
                                start_time = datetime.strptime(
                                    f"{date_str} {day_schedule['start_time']}", 
                                    "%Y-%m-%d %H:%M"
                                ).replace(tzinfo=timezone.utc)
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Failed to parse start_time for {day}: {str(e)}")
                                # Continue without start_time
                        
                        end_time = None
                        if day_schedule.get("end_time") and day_schedule["end_time"]:
                            try:
                                end_time = datetime.strptime(
                                    f"{date_str} {day_schedule['end_time']}", 
                                    "%Y-%m-%d %H:%M"
                                ).replace(tzinfo=timezone.utc)
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Failed to parse end_time for {day}: {str(e)}")
                                # Continue without end_time
                        
                        weekly_schedule_dict[day] = DailySchedule(
                            date=schedule_date,
                            start_time=start_time,
                            end_time=end_time,
                            notes=day_schedule.get("notes"),
                            complete=day_schedule.get("complete", False),
                        ).model_dump()
                    except (KeyError, Exception) as e:
                        logger.warning(f"Failed to parse schedule for {day}: {str(e)}")
                        # Try to use fallback date even if day_schedule is malformed
                        fallback_date = week_dates_fallback.get(day.lower())
                        if fallback_date:
                            try:
                                schedule_date = datetime.strptime(fallback_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                                weekly_schedule_dict[day] = DailySchedule(
                                    date=schedule_date,
                                    start_time=None,
                                    end_time=None,
                                    notes=None,
                                    complete=False,
                                ).model_dump()
                                logger.info(f"Created fallback schedule for {day} with date {fallback_date}")
                            except Exception as fallback_error:
                                logger.error(f"Failed to create fallback schedule for {day}: {str(fallback_error)}")
                                continue
                        else:
                            continue
                
                # Create action item
                action_item = ActionItemCreate(
                    goal_id=goal_id,
                    title=action_item_data["title"],
                    description=action_item_data["description"],
                    user_email=user_email,
                    priority=ActionPriority(action_item_data["priority"]),
                    weekly_schedule=weekly_schedule_dict,
                )
                action_items_to_create.append(action_item)
                
            except (KeyError, ValueError) as e:
                logger.error(f"Failed to create action item: {str(e)}", exc_info=True)
                continue
        
        if not action_items_to_create:
            raise Exception("No action items were successfully created")
        
        # Encrypt all items
        encrypted_items = [
            self.encryption_service.encrypt_document(item, ActionItemCreate)
            for item in action_items_to_create
        ]
        
        # Batch insert all action items at once
        items_to_insert = [item.model_dump() for item in encrypted_items]
        insert_results = await self.action_items_collection.insert_many(items_to_insert)
        
        # Create response objects with IDs (keep encrypted for HIPAA compliance)
        action_items = []
        for i, encrypted_item in enumerate(encrypted_items):
            action_item_response = ActionItem(
                id=str(insert_results.inserted_ids[i]),
                **encrypted_item.model_dump()
            )
            action_items.append(action_item_response)
        
        return action_items

    # TODO: Fix the pillar preferences schema it's not coming well on swagger
    async def generate_goal_plan(
        self,
        goal_id: str,
        user_email: str,
        pillar_preferences: Optional[List[PillarTimePreferences]],
    ) -> dict:
        """Generate action items plan for a goal with optimized latency."""
        try:
            # Step 1: Get goal (fast - ~100ms)
            goal = await self.get_goal_by_id(goal_id, user_email)
            if not goal:
                logger.warning(f"Goal {goal_id} not found for user {user_email}")
                return {"success": False, "message": "Goal not found."}

            goal_text = (
                f"{goal.title} {goal.description or ''} {goal.category or ''}".strip()
            )

            # Step 2: Vector search (fast if no docs - ~50ms)
            context_list = []
            try:
                search_results = self.vector_store.search(
                    query=goal_text, user_email=user_email, top_k=5
                )
                context_list = [
                    doc.get("text", "") for doc in search_results if doc.get("text")
                ]
                if context_list:
                    logger.info(f"Found {len(context_list)} context documents for goal planning")
                else:
                    logger.info("No context documents found - skipping categorization")
            except Exception as e:
                logger.warning(f"Vector search failed, continuing without context: {str(e)}", exc_info=True)
                context_list = []

            # Step 3: Categorize context - SKIP if no context (saves 10-20 seconds!)
            health_context = {
                "medical_context": [],
                "lifestyle_factors": [],
                "risk_factors": [],
                "other_context": [],
            }
            
            if context_list:  # Only categorize if we have context
                try:
                    health_context = await self._invoke_structured_llm(
                        schema=CONTEXT_CATEGORY_SCHEMA,
                        system_prompt="Categorize the following context items into a health goal",
                        user_prompt="Context items: {context_items}",
                        input_vars={"context_items": "\n".join(context_list)},
                    )
                    logger.info("Successfully categorized health context")
                except Exception as e:
                    logger.warning(f"Context categorization failed: {str(e)}", exc_info=True)
                    health_context["lifestyle_factors"] = context_list
            else:
                logger.info("No context found, skipping categorization LLM call")

            # Step 4: Format pillar preferences (fast - ~1ms)
            pillar_pref_str = self._format_pillar_preferences(pillar_preferences)
            
            # Step 4.5: Get current week dates for the prompt
            week_dates_str = self._get_current_week_dates()

            # Step 5: Generate action items (main delay - ~30-50s, unavoidable)
            try:
                action_item_with_schedule = await self._invoke_structured_llm(
                    schema=ACTION_ITEM_SCHEMA,
                    system_prompt=GENERATE_ACTION_PLAN_WITH_SCHEDULE_SYSTEM_PROMPT,
                    user_prompt=GENERATE_ACTION_PLAN_WITH_SCHEDULE_USER_PROMPT,
                    input_vars={
                        "goal_title": goal.title,
                        "goal_description": goal.description or "",
                        "medical_context": "\n".join(health_context.get("medical_context", [])) or "None",
                        "lifestyle_factors": "\n".join(health_context.get("lifestyle_factors", [])) or "None",
                        "risk_factors": "\n".join(health_context.get("risk_factors", [])) or "None",
                        "pillar_preferences": pillar_pref_str,
                        "week_dates": week_dates_str,
                    },
                )
                
                if not action_item_with_schedule.get("action_items"):
                    logger.error("LLM returned no action items")
                    return {"success": False, "message": "Failed to generate action items. Please try again."}
                
                logger.info(f"Generated {len(action_item_with_schedule['action_items'])} action items")
                
                # Step 5.5: VALIDATE AND FIX LLM RESPONSE (GUARDRAILS)
                action_item_with_schedule = self._validate_and_fix_llm_response(
                    action_item_with_schedule, week_dates_str
                )
                
            except Exception as e:
                logger.error(f"Failed to generate action items: {str(e)}", exc_info=True)
                raise Exception(f"Action item generation failed: {str(e)}")
            
            # Step 6: Normalize schedule times (fast - ~1ms)
            action_item_with_schedule = self._normalize_schedule_times(action_item_with_schedule)

            # Step 7: Create and save action items in batch (fast - ~200ms instead of ~500ms per item)
            try:
                action_items = await self._create_and_save_action_items(
                    action_item_with_schedule, goal_id, user_email
                )
                logger.info(f"Successfully saved {len(action_items)} action items")
            except Exception as e:
                logger.error(f"Failed to save action items: {str(e)}", exc_info=True)
                raise Exception(f"Failed to save action items: {str(e)}")

            # Step 8: Generate nudges (non-critical, don't fail on error)
            try:
                from app.services.backend_services.nudge_service import get_nudge_service
                nudge_service = get_nudge_service()
                await nudge_service.create_nudges_from_goal(goal_id)
            except Exception as e:
                logger.error(f"[GOAL_PLAN] Failed to generate nudges (non-critical): {type(e).__name__}: {str(e)}", exc_info=True)

            return {
                "success": True,
                "message": "Goal plan generated successfully",
                "data": {
                    "goal": goal.model_dump(),
                    "action_items": [action_item.model_dump() for action_item in action_items],
                },
            }
        except Exception as e:
            logger.error(f"Critical error generating goal plan: {str(e)}", exc_info=True)
            return {"success": False, "message": f"Failed to generate plan: {str(e)}"}


    def _convert_time_objects_to_str(self, obj: Any) -> Any:
        """Recursively convert time-related objects and ObjectIds to string format"""
        if isinstance(obj, dict):
            return {key: self._convert_time_objects_to_str(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_time_objects_to_str(item) for item in obj]
        elif isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, timedelta):
            total_seconds = int(obj.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, time):
            return obj.strftime("%H:%M:%S")
        return obj

    async def get_goal_plan(self, goal_id: str, user_email: str) -> Optional[Dict[str, Any]]:
        """Get stored action plan and schedule for a goal"""
        action_plan = await self.action_plans_collection.find_one({
            "goal_id": goal_id,
            "user_email": user_email
        })
        weekly_schedule = await self.schedules_collection.find_one({
            "goal_id": goal_id,
            "user_email": user_email
        })
        
        if not action_plan and not weekly_schedule:
            return None

        if action_plan:
            action_plan["id"] = str(action_plan["_id"])
            del action_plan["_id"]
        
        if weekly_schedule:
            weekly_schedule["id"] = str(weekly_schedule["_id"])
            del weekly_schedule["_id"]

        return {
            "action_plan": action_plan,
            "weekly_schedule": weekly_schedule
        }

    def _normalize_weekly_schedule_for_response(self, weekly_schedule: dict) -> dict:
        """Convert weekly schedule data to proper format for WeeklyActionSchedule schema.
        
        Handles:
        - date: string -> datetime
        - start_time: time string ("HH:MM") or empty -> datetime or None
        - end_time: time string ("HH:MM") or empty -> datetime or None
        """
        normalized = {}
        for day_name, day_schedule in weekly_schedule.items():
            if not day_schedule:
                normalized[day_name] = None
                continue
            
            # Parse date
            date_str = day_schedule.get("date")
            if isinstance(date_str, datetime):
                schedule_date = date_str
            elif isinstance(date_str, str):
                # Skip placeholder/template strings like 'YYYY-MM-DD'
                if date_str.upper() in ['YYYY-MM-DD', 'YYYY/MM/DD', 'YYYY_MM_DD']:
                    normalized[day_name] = None
                    continue
                try:
                    schedule_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Invalid date format in schedule normalization: {date_str}")
                    normalized[day_name] = None
                    continue
            else:
                normalized[day_name] = None
                continue
            
            # Parse start_time
            start_time = None
            start_time_str = day_schedule.get("start_time", "")
            if start_time_str and isinstance(start_time_str, str) and start_time_str.strip():
                try:
                    # If it's just time (HH:MM), combine with date
                    if ":" in start_time_str and len(start_time_str.split(":")) == 2:
                        hour, minute = map(int, start_time_str.split(":"))
                        start_time = datetime.combine(schedule_date.date(), time(hour, minute))
                    # If it's already a datetime string, parse it
                    elif "T" in start_time_str or " " in start_time_str:
                        start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass
            elif isinstance(start_time_str, datetime):
                start_time = start_time_str
            
            # Parse end_time
            end_time = None
            end_time_str = day_schedule.get("end_time", "")
            if end_time_str and isinstance(end_time_str, str) and end_time_str.strip():
                try:
                    # If it's just time (HH:MM), combine with date
                    if ":" in end_time_str and len(end_time_str.split(":")) == 2:
                        hour, minute = map(int, end_time_str.split(":"))
                        end_time = datetime.combine(schedule_date.date(), time(hour, minute))
                    # If it's already a datetime string, parse it
                    elif "T" in end_time_str or " " in end_time_str:
                        end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass
            elif isinstance(end_time_str, datetime):
                end_time = end_time_str
            
            normalized[day_name] = DailySchedule(
                date=schedule_date,
                start_time=start_time,
                end_time=end_time,
                notes=day_schedule.get("notes"),
                complete=day_schedule.get("complete", False)
            )
        
        return normalized

    # TODO: You can create a get action_item by id function here 
    async def mark_action_item_complete(self, action_item_id: str, weekday_index: int) -> ActionItem:
        # Validate weekday_index
        if weekday_index < 0 or weekday_index > 6:
            raise ValueError(f"Invalid weekday_index: {weekday_index}. Must be between 0 (Monday) and 6 (Sunday).")
        
        # Validate ObjectId format
        try:
            object_id = ObjectId(action_item_id)
        except Exception:
            raise ValueError(f"Invalid action item ID format: '{action_item_id}'. Must be a valid 24-character MongoDB ObjectId.")
        
        action_item_doc = await self.action_items_collection.find_one({ "_id": object_id })
        if not action_item_doc:
            raise ValueError(f"Action item with ID '{action_item_id}' not found. Make sure you're using an action_item_id (from action item's 'id' field), not a goal_id.")
        
        action_item = self.encryption_service.decrypt_document(action_item_doc, ActionItem)
        
        # Handle both dict and ActionItem instance from decryption
        if isinstance(action_item, ActionItem):
            action_item_dict = action_item.model_dump()
        else:
            action_item_dict = action_item
        
        weekday = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][weekday_index]
        weekly_schedule = action_item_dict.get("weekly_schedule", {})
        
        if not weekly_schedule or weekday not in weekly_schedule:
            raise ValueError(f"Weekday {weekday} not found in action item schedule")
        
        weekly_schedule[weekday]["complete"] = True
        action_item_dict["weekly_schedule"] = weekly_schedule
        
        # Encrypt and update
        encrypted_item = self.encryption_service.encrypt_document(action_item_dict, ActionItem)
        await self.action_items_collection.update_one(
            { "_id": object_id},
            { "$set": { "weekly_schedule": encrypted_item["weekly_schedule"] if isinstance(encrypted_item, dict) else encrypted_item.weekly_schedule } }
        )
        
        # Normalize weekly schedule for response (convert time strings to datetime objects)
        normalized_schedule = self._normalize_weekly_schedule_for_response(weekly_schedule)
        
        # Prepare response
        if isinstance(encrypted_item, dict):
            encrypted_item["id"] = str(action_item_doc["_id"])
            encrypted_item["priority"] = ActionPriority(encrypted_item["priority"])
            encrypted_item["weekly_schedule"] = WeeklyActionSchedule(**normalized_schedule)
            return ActionItem(**encrypted_item)
        else:
            # If it's already an ActionItem, convert for response
            item_dict = encrypted_item.model_dump()
            item_dict["id"] = str(action_item_doc["_id"])
            item_dict["priority"] = ActionPriority(item_dict["priority"])
            item_dict["weekly_schedule"] = WeeklyActionSchedule(**normalized_schedule)
            return ActionItem(**item_dict)
    
    async def mark_action_item_incomplete(self, action_item_id: str, weekday_index: int) -> ActionItem:
        # Validate weekday_index
        if weekday_index < 0 or weekday_index > 6:
            raise ValueError(f"Invalid weekday_index: {weekday_index}. Must be between 0 (Monday) and 6 (Sunday).")
        
        # Validate ObjectId format
        try:
            object_id = ObjectId(action_item_id)
        except Exception:
            raise ValueError(f"Invalid action item ID format: '{action_item_id}'. Must be a valid 24-character MongoDB ObjectId.")
        
        action_item_doc = await self.action_items_collection.find_one({ "_id": object_id })
        if not action_item_doc:
            raise ValueError(f"Action item with ID '{action_item_id}' not found. Make sure you're using an action_item_id (from action item's 'id' field), not a goal_id.")
        
        action_item = self.encryption_service.decrypt_document(action_item_doc, ActionItem)
        
        # Handle both dict and ActionItem instance from decryption
        if isinstance(action_item, ActionItem):
            action_item_dict = action_item.model_dump()
        else:
            action_item_dict = action_item
        
        weekday = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][weekday_index]
        weekly_schedule = action_item_dict.get("weekly_schedule", {})
        
        if not weekly_schedule or weekday not in weekly_schedule:
            raise ValueError(f"Weekday {weekday} not found in action item schedule")
        
        weekly_schedule[weekday]["complete"] = False
        action_item_dict["weekly_schedule"] = weekly_schedule
        
        # Encrypt and update
        encrypted_item = self.encryption_service.encrypt_document(action_item_dict, ActionItem)
        await self.action_items_collection.update_one(
            { "_id": object_id},
            { "$set": { "weekly_schedule": encrypted_item["weekly_schedule"] if isinstance(encrypted_item, dict) else encrypted_item.weekly_schedule } }
        )
        
        # Normalize weekly schedule for response (convert time strings to datetime objects)
        normalized_schedule = self._normalize_weekly_schedule_for_response(weekly_schedule)
        
        # Prepare response
        if isinstance(encrypted_item, dict):
            encrypted_item["id"] = str(action_item_doc["_id"])
            encrypted_item["priority"] = ActionPriority(encrypted_item["priority"])
            encrypted_item["weekly_schedule"] = WeeklyActionSchedule(**normalized_schedule)
            return ActionItem(**encrypted_item)
        else:
            # If it's already an ActionItem, convert for response
            item_dict = encrypted_item.model_dump()
            item_dict["id"] = str(action_item_doc["_id"])
            item_dict["priority"] = ActionPriority(item_dict["priority"])
            item_dict["weekly_schedule"] = WeeklyActionSchedule(**normalized_schedule)
            return ActionItem(**item_dict)


def get_goals_service() -> GoalsService:
    return GoalsService()
