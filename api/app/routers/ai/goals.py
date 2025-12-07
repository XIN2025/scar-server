from fastapi import APIRouter, HTTPException, Query, Body, Depends
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional
from datetime import datetime, timedelta, date
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import Request
from ...schemas.ai.goals import (
    ActionItemCreate, GoalUpdate, Goal, GoalProgressUpdate, GoalNote,
    WeeklyReflection, GoalStats, GoalResponse, GoalCreate, GoalResponse, WeeklyReflectionCreate
)
from ...schemas.backend.preferences import PillarTimePreferences
from ...schemas.backend.action_completions import ActionItemCompletionCreateRequest, ActionItemCompletionUpdate, ActionItemCompletionCreate
from ...services.ai_services.goals_service import GoalsService
from pydantic import EmailStr
from app.services.backend_services.encryption_service import get_encryption_service
from app.core.security import get_current_user, CurrentUser

goals_router = APIRouter()
goals_service = GoalsService()
encryption_service = get_encryption_service()


# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# --- DAILY COMPLETION ENDPOINT (STATIC, PLACE ABOVE DYNAMIC ROUTES) ---
@goals_router.get("/api/goals/daily-completion", response_model=GoalResponse)
async def get_daily_completion(
    month: int = Query(..., ge=1, le=12),
    year: int = Query(..., ge=2000, le=2100),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Return a mapping of YYYY-MM-DD to number of completed action items for the user in the given month/year."""
    try:
        daily_completion = await goals_service.get_current_daily_completion(current_user.email, month, year)
        return GoalResponse(
            success=True,
            message="Daily completion retrieved successfully",
            data={"daily_completion": daily_completion}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@goals_router.get("/api/goals/stats", response_model=GoalResponse)
async def get_goal_stats(current_user: CurrentUser = Depends(get_current_user)):
    try:
        stats = await goals_service.get_goal_stats(current_user.email)
        return GoalResponse(success=True, message="Goal statistics retrieved successfully", data={"stats": stats.dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@goals_router.post("/api/goals/reflection", response_model=GoalResponse)
async def save_weekly_reflection(
    reflection_data: WeeklyReflectionCreate,
    current_user: CurrentUser = Depends(get_current_user)
):
    try:
        # Set user_email from token
        reflection_data.user_email = current_user.email
        result = await goals_service.save_weekly_reflection(reflection_data)
        return GoalResponse(success=True, message="Weekly reflection saved successfully", data=result.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@goals_router.post("/api/goals", response_model=GoalResponse)
async def create_goal(
    goal_data: GoalCreate,
    current_user: CurrentUser = Depends(get_current_user)
):
    try:
        # Set user_email from token
        goal_data.user_email = current_user.email
        goal = await goals_service.create_goal(goal_data)
        return GoalResponse(success=True, message="Goal created successfully", data={"goal": goal.dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@goals_router.get("/api/goals", response_model=GoalResponse)
async def get_user_goals(current_user: CurrentUser = Depends(get_current_user)):
    try:
        goals = await goals_service.get_user_goals(current_user.email)
        # Convert Pydantic models to dicts for JSON serialization
        goals_dict = [goal.model_dump() if hasattr(goal, 'model_dump') else goal for goal in goals]
        return GoalResponse(success=True, message="Goals retrieved successfully", data={"goals": goals_dict})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@goals_router.get("/api/goals/{goal_id}", response_model=GoalResponse)
async def get_goal(goal_id: str, current_user: CurrentUser = Depends(get_current_user)):
    try:
        goal = await goals_service.get_goal_by_id(goal_id, current_user.email)
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")
        return GoalResponse(success=True, message="Goal retrieved successfully", data={"goal": goal.dict()})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# TODO: Update delete_goal function of goal services for new architecture where action_items are store in a separate collection
@goals_router.delete("/api/goals/{goal_id}", response_model=GoalResponse)
async def delete_goal(goal_id: str, current_user: CurrentUser = Depends(get_current_user)):
    try:
        deleted = await goals_service.delete_goal(goal_id, current_user.email)
        if not deleted:
            raise HTTPException(status_code=404, detail="Goal not found")
        return GoalResponse(success=True, message="Goal deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@goals_router.post("/api/goals/{goal_id}/generate-plan", response_model=GoalResponse)
async def generate_goal_plan(
    goal_id: str,
    pillar_preferences: List[PillarTimePreferences] = Body(default=[]),
    current_user: CurrentUser = Depends(get_current_user)
):
    import logging
    logger = logging.getLogger(__name__)
    try:
        # Set user_email in pillar_preferences from token
        for pref in pillar_preferences:
            pref.user_email = current_user.email
        result = await goals_service.generate_goal_plan(
            goal_id,
            current_user.email,
            pillar_preferences
        )
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        return GoalResponse(
            success=True,
            message="Goal plan generated successfully",
            data=result["data"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GOAL_PLAN_ENDPOINT] Error generating goal plan: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@goals_router.post("/api/action-items/{action_item_id}/complete", response_model=GoalResponse)
async def mark_action_item_complete(
    action_item_id: str,
    weekday_index: int = Body(..., embed=True),
):
    try:
        completion = await goals_service.mark_action_item_complete(action_item_id, weekday_index)
        return GoalResponse(
            success=True,
            message="Action item marked completed successfully",
            data={"completion": completion.model_dump()}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@goals_router.post("/api/action-items/{action_item_id}/incomplete", response_model=GoalResponse)
async def mark_action_item_incomplete(
    action_item_id: str,
    weekday_index: int = Body(..., embed=True),
):
    try:
        completion = await goals_service.mark_action_item_incomplete(action_item_id, weekday_index)
        return GoalResponse(
            success=True,
            message="Action item marked incomplete successfully",
            data={"completion": completion.model_dump()}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# TODO: Move the logic for this into goal services
@goals_router.get("/api/goals/{goal_id}/action-items", response_model=GoalResponse)
async def get_goal_action_items(
    goal_id: str,
):
    try:
        action_items = await goals_service.action_items_collection.find({
            "goal_id": goal_id,
        }).to_list(None)
        action_items = encryption_service.decrypt_documents_bulk(action_items, schema_class=ActionItemCreate)

        if not action_items:
            raise HTTPException(status_code=404, detail="Action items not found")

        for action_item in action_items:
            if "_id" in action_item:
                action_item["id"] = str(action_item["_id"])
                del action_item["_id"]

        return GoalResponse(
            success=True,
            message="Action items retrieved successfully",
            data={"action_items": action_items}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



### just for debugging goals notifications needs to be cleaned up later
@goals_router.post("/api/goals/test-now", response_model=GoalResponse)
async def create_test_goal_for_now(current_user: CurrentUser = Depends(get_current_user)):
    """Create a test goal with action items scheduled for 1:45 AM today to test notifications immediately"""
    from bson import ObjectId
    from datetime import datetime, timezone, timedelta
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        user_email = current_user.email
        logger.info(f"[TEST_GOAL] Creating test goal for {user_email} scheduled for RIGHT NOW")
        
        now = datetime.now(timezone.utc)
        target_time = now + timedelta(minutes=20)
        reminder_time = target_time - timedelta(minutes=10)
        
        logger.info(f"[TEST_GOAL] Current time: {now}")
        logger.info(f"[TEST_GOAL] Target end_time: {target_time} (20 minutes from now)")
        logger.info(f"[TEST_GOAL] Reminder will be at: {reminder_time} (10 minutes from now)")
        
        goal_id = str(ObjectId())
        goal_data = GoalCreate(
            user_email=user_email,
            title="Test Goal for Immediate Notification",
            description="This is a test goal to verify notifications work",
            priority="medium",
            category="health"
        )
        
        goal = await goals_service.create_goal(goal_data)
        logger.info(f"[TEST_GOAL] Created goal: {goal.id}")
        
        from app.schemas.ai.goals import ActionItemCreate, WeeklyActionSchedule, DailySchedule, ActionPriority
        from app.services.backend_services.encryption_service import get_encryption_service
        enc_service = get_encryption_service()
        
        weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        today_weekday = weekday_names[target_time.weekday()]
        
        daily_schedule = DailySchedule(
            date=target_time.replace(hour=0, minute=0, second=0, microsecond=0),
            start_time=target_time - timedelta(minutes=15),
            end_time=target_time,
            notes="Test action item for immediate notification testing",
            complete=False
        )
        
        weekly_schedule_dict = {day: None for day in weekday_names}
        weekly_schedule_dict[today_weekday] = daily_schedule
        
        action_item = ActionItemCreate(
            goal_id=goal.id,
            user_email=user_email,
            title="Test Action Item - Immediate",
            description="This action item is scheduled for right now to test notifications",
            priority=ActionPriority.HIGH,
            weekly_schedule=WeeklyActionSchedule(**weekly_schedule_dict)
        )
        
        action_item_dict = enc_service.encrypt_document(action_item, ActionItemCreate).model_dump()
        action_item_dict["_id"] = ObjectId()
        action_item_dict["created_at"] = datetime.now(timezone.utc)
        
        result = await goals_service.action_items_collection.insert_one(action_item_dict)
        logger.info(f"[TEST_GOAL] Created action item: {result.inserted_id}")
        
        from app.services.backend_services.nudge_service import get_nudge_service
        nudge_service = get_nudge_service()
        nudges = await nudge_service.create_nudges_from_goal(goal.id)
        logger.info(f"[TEST_GOAL] Created {len(nudges)} nudges")
        
        return GoalResponse(
            success=True,
            message=f"Test goal created with action item scheduled for {target_time}. Reminder at {reminder_time}",
            data={
                "goal_id": goal.id,
                "action_item_id": str(result.inserted_id),
                "target_time": target_time.isoformat(),
                "reminder_time": reminder_time.isoformat(),
                "nudges_created": len(nudges)
            }
        )
    except Exception as e:
        logger.error(f"[TEST_GOAL] Error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
