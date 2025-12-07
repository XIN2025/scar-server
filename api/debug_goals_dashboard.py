"""
Debug script to check why nutrition goals aren't showing on dashboard
"""
import asyncio
import sys
import os
from datetime import datetime, timezone
from bson import ObjectId

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.db import get_db
from app.services.backend_services.encryption_service import get_encryption_service
from app.schemas.ai.goals import Goal, ActionItem

async def debug_goals(user_email: str):
    """Debug goals and action items for a user"""
    db = get_db()
    encryption_service = get_encryption_service()
    
    goals_collection = db["goals"]
    action_items_collection = db["action_items"]
    
    print(f"\n=== Debugging Goals for: {user_email} ===\n")
    
    # Get all goals
    goals = await goals_collection.find({"user_email": user_email}).to_list(None)
    goals = encryption_service.decrypt_documents_bulk(goals, Goal)
    
    print(f"Total Goals Found: {len(goals)}\n")
    
    for goal in goals:
        goal_id = str(goal["_id"])
        print(f"Goal ID: {goal_id}")
        print(f"  Title: {goal.get('title', 'N/A')}")
        print(f"  Category: {goal.get('category', 'N/A')}")
        print(f"  Created: {goal.get('created_at', 'N/A')}")
        
        # Get action items for this goal
        action_items = await action_items_collection.find({
            "goal_id": goal_id
        }).to_list(None)
        action_items = encryption_service.decrypt_documents_bulk(action_items, ActionItem)
        
        print(f"  Action Items: {len(action_items)}")
        
        for idx, item in enumerate(action_items):
            print(f"\n  Action Item {idx + 1}:")
            print(f"    Title: {item.get('title', 'N/A')}")
            weekly_schedule = item.get("weekly_schedule", {})
            
            if not weekly_schedule:
                print(f"    ⚠️  WARNING: No weekly_schedule found!")
                continue
            
            # Check each day
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            for day in days:
                day_schedule = weekly_schedule.get(day)
                if day_schedule:
                    date_val = day_schedule.get("date")
                    if isinstance(date_val, datetime):
                        date_str = date_val.strftime("%Y-%m-%d")
                    elif isinstance(date_val, str):
                        date_str = date_val
                    else:
                        date_str = str(date_val)
                    
                    complete = day_schedule.get("complete", False)
                    start_time = day_schedule.get("start_time")
                    end_time = day_schedule.get("end_time")
                    
                    print(f"    {day.capitalize()}: date={date_str}, complete={complete}, start={start_time}, end={end_time}")
                else:
                    print(f"    {day.capitalize()}: ❌ NOT SET")
        
        print("\n" + "="*60 + "\n")
    
    # Check today's date
    today = datetime.now(timezone.utc)
    weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    today_weekday = weekday_names[today.weekday()]
    print(f"Today is: {today.strftime('%Y-%m-%d')} ({today_weekday})")
    print(f"\nIf dashboard filters by today, it will look for {today_weekday} schedules with date={today.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_goals_dashboard.py <user_email>")
        sys.exit(1)
    
    user_email = sys.argv[1]
    asyncio.run(debug_goals(user_email))

