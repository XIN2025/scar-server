from .db import get_db
import httpx
from datetime import datetime, timedelta, timezone
from bson import ObjectId
try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None
from app.schemas.backend.nudge import Nudge, NudgeType, NudgeStatus, GoalNudge
from app.exceptions import (
    NotificationError,
    GoalNotFoundError,
    UserNotFoundError,
    TokenNotFoundError,
    NotificationDisabledError,
)
from typing import List, Dict, Any, Optional
import firebase_admin
from firebase_admin import credentials, messaging
from firebase_admin.exceptions import InvalidArgumentError, NotFoundError
import os
from app.services.backend_services.encryption_service import get_encryption_service
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from dotenv import load_dotenv
from app.prompts import GENERATE_MORNING_NOTIFICATION_PROMPT, GENERATE_EVENING_NOTIFICATION_PROMPT, GENERATE_NIGHT_NOTIFICATION_PROMPT, GENERATE_CHECKIN_NOTIFICATION_PROMPT
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.schemas.ai.goals import Goal, ActionItem, GoalWithActionItems
from app.config import OPENAI_API_KEY, LLM_MODEL, MONGODB_URI, DB_NAME, NUDGE_SCHEDULED_COLLECTION, FIREBASE_SERVICE_ACCOUNT_PATH_ENV, DEFAULT_FIREBASE_SERVICE_ACCOUNT_PATH
import asyncio
import logging
from zoneinfo import ZoneInfo

load_dotenv()
logger = logging.getLogger(__name__)

async def send_fcm_notification_job(email, title, body):
    import uuid
    execution_id = str(uuid.uuid4())[:8]
    try:
        nudge_service = get_nudge_service()
        await nudge_service.send_fcm_notification(email, title, body, execution_id=execution_id)
    except Exception as e:
        logger.error(f"[JOB] [{execution_id}] Error in goal reminder notification job for {email}: {type(e).__name__}: {str(e)}", exc_info=True)
        raise

def run_async_job(coro, *args, **kwargs):
    loop = asyncio.get_event_loop()
    loop.create_task(coro(*args, **kwargs))

async def morning_notification_job(email):
    import uuid
    execution_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
   
    nudge_service = get_nudge_service()
   
    try:
        logger.info(f"[JOB] [{execution_id}] Morning notification job triggered by scheduler for {email} at {timestamp}")
        print(f"[JOB] [{execution_id}] 📅 Scheduler triggered morning notification for {email}...")
        await nudge_service.send_morning_notification(email, execution_id=execution_id)
    except UserNotFoundError as e:
        logger.warning(f"Cannot send morning notification: User not found: {email}")
    except TokenNotFoundError as e:
        logger.warning(f"Cannot send morning notification: No FCM token registered for {email}")
    except NotificationDisabledError as e:
        logger.info(f"Cannot send morning notification: Notifications disabled for {email}")
    except NotificationError as e:
        logger.warning(f"Failed to send morning notification to {email}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error sending morning notification to {email}: {type(e).__name__}: {str(e)}", exc_info=True)

async def evening_notification_job(email):
    import uuid
    execution_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
   
    nudge_service = get_nudge_service()
   
    try:
        logger.info(f"[JOB] [{execution_id}] Evening notification job triggered by scheduler for {email} at {timestamp}")
        print(f"[JOB] [{execution_id}] 📅 Scheduler triggered evening notification for {email}...")
        await nudge_service.send_evening_notification(email, execution_id=execution_id)
    except UserNotFoundError as e:
        logger.warning(f"Cannot send evening notification: User not found: {email}")
    except TokenNotFoundError as e:
        logger.warning(f"Cannot send evening notification: No FCM token registered for {email}")
    except NotificationDisabledError as e:
        logger.info(f"Cannot send evening notification: Notifications disabled for {email}")
    except NotificationError as e:
        logger.warning(f"Failed to send evening notification to {email}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error sending evening notification to {email}: {type(e).__name__}: {str(e)}", exc_info=True)

async def night_notification_job(email):
    import uuid
    execution_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
   
    nudge_service = get_nudge_service()
   
    try:
        logger.info(f"[JOB] [{execution_id}] Night notification job triggered by scheduler for {email} at {timestamp}")
        print(f"[JOB] [{execution_id}] 📅 Scheduler triggered night notification for {email}...")
        await nudge_service.send_night_notification(email, execution_id=execution_id)
    except UserNotFoundError as e:
        logger.warning(f"Cannot send night notification: User not found: {email}")
    except TokenNotFoundError as e:
        logger.warning(f"Cannot send night notification: No FCM token registered for {email}")
    except NotificationDisabledError as e:
        logger.info(f"Cannot send night notification: Notifications disabled for {email}")
    except NotificationError as e:
        logger.warning(f"Failed to send night notification to {email}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error sending night notification to {email}: {type(e).__name__}: {str(e)}", exc_info=True)

async def checkin_notification_job(email):
    import uuid
    execution_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
   
    nudge_service = get_nudge_service()
   
    try:
        logger.info(f"[NOTIFICATION] [{execution_id}] Check-in notification job triggered for {email} at {timestamp}")
        print(f"[{execution_id}] Sending check-in notification for {email}...")
        await nudge_service.send_checkin_notification(email, execution_id=execution_id)
    except UserNotFoundError as e:
        logger.warning(f"Cannot send check-in notification: User not found: {email}")
    except TokenNotFoundError as e:
        logger.warning(f"Cannot send check-in notification: No FCM token registered for {email}")
    except NotificationDisabledError as e:
        logger.info(f"Cannot send check-in notification: Notifications disabled for {email}")
    except NotificationError as e:
        logger.warning(f"Failed to send check-in notification to {email}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error sending check-in notification to {email}: {type(e).__name__}: {str(e)}", exc_info=True)

class NudgeService:
    def __init__(self):
        print(f"[DEBUG] Initializing NudgeService at {datetime.now()}")
        self.db = get_db()
        self.users_collection = self.db["users"]
        self.goals_collection = self.db["goals"]
        self.action_items_collection = self.db["action_items"]
        self.nudges_collection = self.db["nudges"]
        self.encryption_service = get_encryption_service()
        self._initialize_firebase()
        self._jobstores = None
        self._jobdefaults = {'misfire_grace_time': 15 * 60}
        self.scheduler = None
        self._scheduling_lock = asyncio.Lock()
        
        self._locks_collection = self.db["notification_locks"]
        self._create_lock_index()
    
    def _create_lock_index(self):
        try:
            try:
                self._locks_collection.create_index(
                    [("fcm_token", 1), ("type", 1), ("date", 1)],
                    unique=True,
                    name="device_notification_lock"
                )
                logger.info("Created device_notification_lock index")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower() or "IndexOptionsConflict" in str(e):
                    logger.info("device_notification_lock index already exists, skipping")
                else:
                    raise
            try:
                self._locks_collection.create_index(
                    "created_at",
                    expireAfterSeconds=86400,
                    name="ttl_notification_lock"
                )
                logger.info("Created ttl_notification_lock index")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower() or "IndexOptionsConflict" in str(e):
                    logger.info("ttl_notification_lock index already exists (different name), skipping")
                else:
                    raise
            logger.info("Notification lock indexes check completed")
        except Exception as e:
            logger.warning(f"Could not create lock indexes: {e}", exc_info=True)

    def _initialize_scheduler(self):
        # TEMPORARILY DISABLED: Scheduler initialization fails silently since notifications are disabled
        try:
            if self.scheduler is None:
                if self._jobstores is None:
                    try:
                        logger.info(f"[SCHEDULER_INIT] Initializing MongoDBJobStore with database='{DB_NAME}', collection='{NUDGE_SCHEDULED_COLLECTION}', host='{MONGODB_URI[:50]}...'")
                        self._jobstores = {
                            'default': MongoDBJobStore(
                                database=DB_NAME,
                                collection=NUDGE_SCHEDULED_COLLECTION,
                                host=MONGODB_URI
                            )
                        }
                        logger.info(f"[SCHEDULER_INIT] ✓ MongoDBJobStore initialized successfully")
                    except Exception as e:
                        logger.warning(f"[SCHEDULER_INIT] Failed to initialize MongoDBJobStore (notifications disabled): {type(e).__name__}: {str(e)}")
                        # Don't raise - allow service to continue without scheduler
                        return
               
                logger.info(f"[SCHEDULER_INIT] Creating AsyncIOScheduler with jobstores")
                self.scheduler = AsyncIOScheduler(
                    jobstores=self._jobstores,
                    jobdefaults=self._jobdefaults,
                    timezone=ZoneInfo("UTC")
                )
                logger.info(f"[SCHEDULER_INIT] ✓ AsyncIOScheduler created")
        except Exception as e:
            logger.warning(f"[SCHEDULER_INIT] Scheduler initialization failed (notifications disabled): {type(e).__name__}: {str(e)}")
            # Don't raise - allow service to continue without scheduler

    def start_scheduler(self):
        # TEMPORARILY DISABLED: Scheduler is disabled since notifications are disabled
        if self.scheduler is None:
            self._initialize_scheduler()
        
        if self.scheduler is None:
            logger.info("[SCHEDULER] Scheduler not available (notifications disabled), skipping start")
            return
        
        if not self.scheduler.running:
            try:
                logger.info("Starting notification scheduler...")
                self.scheduler.start()
                logger.info("Notification scheduler started successfully")
            except Exception as e:
                logger.warning(f"[SCHEDULER] Failed to start scheduler (notifications disabled): {str(e)}")
        else:
            logger.debug("Scheduler is already running, skipping start")
   
    def stop_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.shutdown()
   
    async def _try_claim_notification_slot(self, fcm_token: str, notification_type: str, email_debug: str = None) -> bool:
        """
        Updated: Claims a slot based on FCM Token instead of Email.
        This prevents the same device from receiving multiple notifications for different accounts.
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        
        try:
            result = await self._locks_collection.insert_one({
                "fcm_token": fcm_token,
                "type": notification_type,
                "date": today,
                "created_at": now,
                "claimed_by_email": email_debug 
            })
            if result.inserted_id:
                logger.info(f"[LOCK] ✓ Claimed slot for {notification_type} on device {fcm_token[:10]}... (User: {email_debug})")
                return True
            else:
                logger.warning(f"[LOCK] ⚠️ Failed to claim slot for {notification_type} on device {fcm_token[:10]}...")
                return False
            
        except Exception as e:
            error_msg = str(e)
            if "E11000" in error_msg or "duplicate" in error_msg.lower() or "duplicate key" in error_msg.lower():
                logger.info(f"[LOCK] ⚠️ Duplicate {notification_type} for device {fcm_token[:10]}... blocked (already claimed today)")
                return False
            logger.error(f"[LOCK] ❌ Error claiming slot for {notification_type} on device: {e}", exc_info=True)
            return False

    async def is_active_user_on_device(self, email: str) -> bool:
        current_user = await self.users_collection.find_one({"email": email})
        if not current_user or not current_user.get("fcm_token"):
            return False 

        token = current_user["fcm_token"]

        users_on_device = await self.users_collection.find(
            {"fcm_token": token},
            {"email": 1, "last_active_at": 1, "updated_at": 1, "created_at": 1}
        ).to_list(length=None)

        if not users_on_device:
            return False

        def get_sort_key(u):
            if u.get("last_active_at"):
                return u["last_active_at"]
            if u.get("updated_at"):
                return u["updated_at"]
            return u.get("created_at", datetime.min)

        users_on_device.sort(key=get_sort_key, reverse=True)

        winner_email = users_on_device[0]["email"]
        
        is_winner = (winner_email == email)
        if not is_winner:
            logger.debug(f"[MULTI-ACCOUNT] {email} is NOT the active user on device. Active is: {winner_email}")
        
        return is_winner

    async def local_time_to_utc(self, local_dt: datetime, email: str) -> datetime:
        user = await self.users_collection.find_one({"email": email})
        if not user or "timezone" not in user:
            raise UserNotFoundError(f"User not found or timezone not set: {email}")
        tz_name = str(user["timezone"]).replace(" ", "_")
        try:
            tz = ZoneInfo(tz_name)
        except Exception as e:
            raise ValueError(f"Invalid IANA timezone '{tz_name}' for user {email}") from e
        if local_dt.tzinfo is None:
            aware_local = local_dt.replace(tzinfo=tz)
        else:
            aware_local = local_dt.astimezone(tz)
        return aware_local.astimezone(ZoneInfo("UTC"))

    async def _invoke_structured_llm(
        self, schema: dict, system_prompt: str, user_prompt: str, input_vars: dict
    ) -> dict:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY
        ).with_structured_output(schema)
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt)]
        )
        chain = prompt | llm
        return await chain.ainvoke(input_vars)

    def _initialize_firebase(self):
        # TEMPORARILY DISABLED: Firebase initialization fails silently since notifications are disabled
        try:
            if not firebase_admin._apps:
                firebase_service_account_path = None
                if FIREBASE_SERVICE_ACCOUNT_PATH_ENV and os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH_ENV):
                    firebase_service_account_path = FIREBASE_SERVICE_ACCOUNT_PATH_ENV
                elif os.path.exists(DEFAULT_FIREBASE_SERVICE_ACCOUNT_PATH):
                    firebase_service_account_path = DEFAULT_FIREBASE_SERVICE_ACCOUNT_PATH
               
                if firebase_service_account_path:
                    try:
                        cred = credentials.Certificate(firebase_service_account_path)
                        firebase_admin.initialize_app(cred)
                        logger.info("Firebase initialized successfully.")
                    except Exception as e:
                        logger.warning(
                            f"[FIREBASE_INIT] Firebase initialization failed (notifications disabled): {str(e)}. "
                            "Continuing without Firebase since notifications are temporarily disabled."
                        )
                else:
                    env_msg = f"Env var path ({FIREBASE_SERVICE_ACCOUNT_PATH_ENV})" if FIREBASE_SERVICE_ACCOUNT_PATH_ENV else "No env var set"
                    default_msg = f"Default path ({DEFAULT_FIREBASE_SERVICE_ACCOUNT_PATH})"
                    logger.warning(
                        f"[FIREBASE_INIT] Firebase service account file not found (notifications disabled). "
                        f"{env_msg} and {default_msg} both missing or invalid. "
                        "Continuing without Firebase since notifications are temporarily disabled."
                    )
        except Exception as e:
            logger.warning(
                f"[FIREBASE_INIT] Firebase initialization error (notifications disabled): {str(e)}. "
                "Continuing without Firebase since notifications are temporarily disabled."
            )

    def _validate_fcm_token_format(self, token: str) -> tuple[bool, str]:
        if not token:
            return False, "Token cannot be empty"
       
        token = token.strip()
       
        if len(token) < 100:
            return False, f"Token too short. Real FCM tokens are typically 150+ characters (got {len(token)})"
       
        if len(token) > 2000:
            return False, f"Token too long (max 2000 chars, got {len(token)})"
       
        invalid_chars = set(token) - set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_:/=.+")
        if invalid_chars:
            return False, f"Token contains invalid characters: {''.join(sorted(invalid_chars))}"
       
        return True, "Valid format"
   
    async def save_fcm_token(self, email: str, fcm_token: str) -> bool:
        fcm_token = fcm_token.strip()
       
        is_valid, error_msg = self._validate_fcm_token_format(fcm_token)
        if not is_valid:
            logger.warning(f"Invalid FCM token format for {email}: {error_msg}")
            return False
       
        user = await self.users_collection.find_one({"email": email})
        if not user:
            logger.warning(f"Cannot register FCM token: User not found: {email}")
            return False
       
        result = await self.users_collection.update_one(
            {"email": email},
            {
                "$set": {
                    "fcm_token": fcm_token, 
                    "updated_at": datetime.now(timezone.utc),
                    "last_active_at": datetime.now(timezone.utc)
                }
            },
            upsert=False,
        )
       
        if result.matched_count == 0:
            logger.warning(f"FCM token update failed: User not found: {email}")
            return False
       
        if result.modified_count > 0:
            logger.info(f"FCM token updated successfully for {email} (length: {len(fcm_token)})")
        else:
            logger.debug(f"FCM token unchanged for {email} (token may be the same)")
       
        return result.acknowledged and result.matched_count > 0

    async def send_fcm_notification(self, email: str, title: str, body: str, fcm_client=None, execution_id: str = None):
        # TEMPORARILY DISABLED: Notifications are disabled
        logger.info(f"[FCM_SEND] Notifications temporarily disabled - skipping notification to {email} | Title: '{title}'")
        return
        # Original code below (temporarily disabled)
        # timestamp = datetime.now(timezone.utc).isoformat()
        # exec_id = execution_id or "unknown"
        # logger.info(f"[FCM_SEND] [{exec_id}] Preparing to send FCM notification to {email} | Title: '{title}' | Body: '{body[:50]}...' | Time: {timestamp}")
       
        user_doc = await self.users_collection.find_one({"email": email})
       
        if not user_doc:
            logger.warning(f"[FCM_SEND] [{exec_id}] User not found: {email}")
            raise UserNotFoundError(f"User not found: {email}")
       
        fcm_token = user_doc.get("fcm_token")
        if not fcm_token or not fcm_token.strip():
            logger.warning(f"[FCM_SEND] [{exec_id}] No FCM token found for user: {email}")
            raise TokenNotFoundError(f"No FCM token found for user: {email}")
            
        if not user_doc.get("notifications_enabled", True):
            logger.info(f"[FCM_SEND] [{exec_id}] Notifications disabled for user: {email}")
            raise NotificationDisabledError(f"Notifications are disabled for user: {email}")
            
        fcm_token = fcm_token.strip()
        
        message = messaging.Message(
            token=fcm_token,
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            data={
                "type": "nudge",
                "email": email,
            },
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id="default",
                    sound="default",
                    color="#16A34A",
                    icon="@drawable/notification_icon",
                ),
            ),
            apns=messaging.APNSConfig(
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(
                        alert=messaging.ApsAlert(
                            title=title,
                            body=body
                        ),
                        sound="default",
                        badge=1
                    )
                ),
                headers={
                    "apns-priority": "10"
                }
            ),
        )
        
        try:
            send_timestamp = datetime.now(timezone.utc).isoformat()
            logger.info(f"[FCM_SEND] [{exec_id}] Calling Firebase messaging.send() for {email} at {send_timestamp}")
            message_id = await asyncio.to_thread(messaging.send, message)
            success_timestamp = datetime.now(timezone.utc).isoformat()
            logger.info(f"[FCM_SEND] [{exec_id}] ✓ FCM notification sent successfully to {email} | Message ID: {message_id} | Time: {success_timestamp} | Title: '{title}'")
            print(f"[{exec_id}] ✓ FCM notification sent to {email} (Message ID: {message_id})")
            return {"success": True, "message_id": message_id}
        except InvalidArgumentError as e:
            error_msg = str(e)
            logger.warning(
                f"Invalid FCM token for {email}: {error_msg}."
            )
            try:
                await self.users_collection.update_one(
                    {"email": email},
                    {"$unset": {"fcm_token": ""}}
                )
                logger.info(f"Cleared invalid FCM token for {email}")
            except Exception as clear_error:
                logger.warning(f"Failed to clear invalid token for {email}: {str(clear_error)}")
           
            raise NotificationError(
                f"Invalid FCM registration token for user {email}. Token may be expired or invalid.",
                details={"email": email, "error_type": "invalid_token"},
            )
        except NotFoundError as e:
            logger.warning(f"FCM token not found in Firebase for {email}: {str(e)}")
            raise NotificationError(
                f"FCM token not found for user {email}",
                details={"email": email, "error_type": "token_not_found"},
            )
        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                f"Unexpected Firebase error for {email}: {error_type}: {str(e)}",
                exc_info=True
            )
            raise NotificationError(
                f"Failed to send FCM notification: {str(e)}",
                details={"email": email, "error_type": error_type},
            )

    async def get_goal_by_id(self, goal_id: str) -> GoalWithActionItems:
        goal = await self.goals_collection.find_one({"_id": ObjectId(goal_id)})
        if not goal:
            return None
        action_items = await self.action_items_collection.find(
            {"goal_id": goal_id}
        ).to_list(length=None)
        goal["action_items"] = [ActionItem(
            id=str(action_item["_id"]),
            goal_id=action_item["goal_id"],
            user_email=action_item["user_email"],
            title=action_item["title"],
            description=action_item["description"],
            priority=action_item["priority"],
            weekly_schedule=action_item["weekly_schedule"],
        ) for action_item in action_items]
        goal = GoalWithActionItems(
            id=str(goal["_id"]),
            user_email=goal["user_email"],
            title=goal["title"],
            description=goal["description"],
            priority=goal["priority"],
            category=goal["category"],
            created_at=goal["created_at"],
            action_items=goal["action_items"],
        )
        goal = self.encryption_service.decrypt_document(goal, GoalWithActionItems)
        return goal
   
    async def _create_reminder_nudges(self, goal_id: str) -> List[GoalNudge]:
        BUFFER_MINUTES = 10
        goal = await self.get_goal_by_id(goal_id)
        if not goal:
            logger.error(f"[REMINDER_NUDGES] Goal not found: {goal_id}", exc_info=True)
            raise GoalNotFoundError(f"Goal not found: {goal_id}")
        nudges = []
        for action_item in goal.action_items:
            if not action_item.weekly_schedule:
                continue
            try:
                if hasattr(action_item.weekly_schedule, 'model_dump'):
                    weekly_schedule_dict = action_item.weekly_schedule.model_dump()
                elif isinstance(action_item.weekly_schedule, dict):
                    weekly_schedule_dict = action_item.weekly_schedule
                else:
                    logger.error(f"[REMINDER_NUDGES] Unexpected weekly_schedule type: {type(action_item.weekly_schedule)}", exc_info=True)
                    continue
            except Exception as e:
                logger.error(f"[REMINDER_NUDGES] Failed to get weekly_schedule dict: {type(e).__name__}: {str(e)}", exc_info=True)
                continue
            for day, daily_schedule in weekly_schedule_dict.items():
                if not daily_schedule:
                    continue
               
                end_time_value = daily_schedule.get("end_time")
                date_value = daily_schedule.get("date")
               
                if not end_time_value or not date_value:
                    logger.warning(f"[REMINDER_NUDGES] Day '{day}' missing end_time or date - SKIPPING")
                    continue
               
                if isinstance(date_value, str):
                    try:
                        date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        if date_parser:
                            try:
                                date = date_parser.parse(date_value)
                            except ValueError as e2:
                                logger.error(f"[REMINDER_NUDGES] Could not parse date: {date_value}, error: {e2}", exc_info=True)
                                continue
                        else:
                            logger.error(f"[REMINDER_NUDGES] Could not parse date: {date_value} (dateutil not available)", exc_info=True)
                            continue
                elif isinstance(date_value, datetime):
                    date = date_value
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=ZoneInfo("UTC"))
                else:
                    logger.error(f"[REMINDER_NUDGES] Unexpected date type: {type(date_value)} for value: {date_value}", exc_info=True)
                    continue
               
                if isinstance(end_time_value, datetime):
                    if end_time_value.tzinfo is None:
                        end_datetime = end_time_value.replace(tzinfo=ZoneInfo("UTC"))
                    else:
                        end_datetime = end_time_value
                elif isinstance(end_time_value, str):
                    try:
                        from datetime import time as dt_time
                        time_parts = end_time_value.split(':')
                        if len(time_parts) >= 2:
                            hour = int(time_parts[0])
                            minute = int(time_parts[1])
                            second = int(time_parts[2]) if len(time_parts) > 2 else 0
                            end_time_obj = dt_time(hour, minute, second)
                            end_datetime = datetime.combine(date.date(), end_time_obj)
                            if date.tzinfo:
                                end_datetime = end_datetime.replace(tzinfo=date.tzinfo)
                        else:
                            end_datetime = datetime.fromisoformat(end_time_value.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        if date_parser:
                            try:
                                end_datetime = date_parser.parse(end_time_value)
                            except ValueError as e2:
                                logger.error(f"[REMINDER_NUDGES] Could not parse end_time: {end_time_value}, error: {e2}", exc_info=True)
                                continue
                        else:
                            logger.error(f"[REMINDER_NUDGES] Could not parse end_time: {end_time_value} (dateutil not available)", exc_info=True)
                            continue
                else:
                    logger.error(f"[REMINDER_NUDGES] Unexpected end_time type: {type(end_time_value)} for value: {end_time_value}", exc_info=True)
                    continue
               
                reminder_time = end_datetime - timedelta(minutes=BUFFER_MINUTES)
                now_utc = datetime.now(timezone.utc)
               
                if reminder_time <= now_utc:
                    continue
               
                nudge = GoalNudge(
                    goal_id=goal_id,
                    user_email=goal.user_email,
                    action_item_title=action_item.title,
                    scheduled_time=reminder_time,
                    title=f"Reminder: {action_item.title}",
                    body=f"You have '{action_item.title}' ending in {BUFFER_MINUTES} minutes for your goal: {goal.title}",
                    status=NudgeStatus.PENDING
                )
                nudges.append(nudge)
        return nudges

    async def create_nudges_from_goal(self, goal_id: str) -> List[GoalNudge]:
        nudges = await self._create_reminder_nudges(goal_id)
       
        if self.scheduler is None:
            self._initialize_scheduler()
       
        if not self.scheduler.running:
            logger.error(f"[CREATE_NUDGES] Scheduler is NOT running. Nudges will not be sent until scheduler starts.")
       
        now = datetime.now(timezone.utc)
        scheduled_count = 0
        skipped_count = 0
        error_count = 0
       
        for nudge in nudges:
            try:
                if nudge.scheduled_time.tzinfo is None:
                    local_utc = nudge.scheduled_time.replace(tzinfo=ZoneInfo("UTC"))
                elif nudge.scheduled_time.tzinfo == ZoneInfo("UTC"):
                    local_utc = nudge.scheduled_time
                else:
                    local_utc = nudge.scheduled_time.astimezone(ZoneInfo("UTC"))
               
                if local_utc < now:
                    skipped_count += 1
                    continue
               
                self.schedule_notification(
                    email=nudge.user_email,
                    title=nudge.title,
                    body=nudge.body,
                    run_datetime=local_utc
                )
                scheduled_count += 1
            except UserNotFoundError as e:
                logger.error(f"[CREATE_NUDGES] UserNotFoundError scheduling nudge for {nudge.user_email}: {str(e)}", exc_info=True)
                error_count += 1
            except Exception as e:
                logger.error(f"[CREATE_NUDGES] Exception scheduling nudge for {nudge.user_email}: {type(e).__name__}: {str(e)}", exc_info=True)
                error_count += 1
       
        logger.info(f"[CREATE_NUDGES] Goal {goal_id}: Total={len(nudges)}, Scheduled={scheduled_count}, skipped={skipped_count}, Errors={error_count}")
        return nudges

    def schedule_notification(self, email: str, title: str, body: str, run_datetime: datetime):
        if self.scheduler is None:
            self._initialize_scheduler()
        job_id = f"nudge_{email}_{run_datetime.isoformat()}"
        try:
            self.scheduler.add_job(
                send_fcm_notification_job,
                trigger='date',
                run_date=run_datetime,
                args=[email, title, body],
                id=job_id,
                replace_existing=True,
                coalesce=True,
            )
        except Exception as e:
            logger.error(f"[SCHEDULE_NOTIFICATION] Failed to schedule notification for {email} at {run_datetime}: {type(e).__name__}: {str(e)}", exc_info=True)
            raise
   
    async def get_goals_summary(self, user_email: str) -> str:
        goals = await self.goals_collection.find({"user_email": user_email}).to_list(None)
        goals_dicts = [self.encryption_service.decrypt_document(item, Goal) for item in goals]
        if not goals_dicts:
            return "No goals found for this user."
        summary_lines = []
        for goal_dict in goals_dicts:
            title = goal_dict.get("title", "Untitled Goal") if isinstance(goal_dict, dict) else getattr(goal_dict, "title", "Untitled Goal")
            description = goal_dict.get("description", "") if isinstance(goal_dict, dict) else getattr(goal_dict, "description", "")
            category = goal_dict.get("category", "General") if isinstance(goal_dict, dict) else getattr(goal_dict, "category", "General")
            completed = goal_dict.get("completed", False) if isinstance(goal_dict, dict) else getattr(goal_dict, "completed", False)
            status = "Completed" if completed else "In Progress"
            summary_lines.append(f"- {title} ({category}): {description} [{status}]")
        summary = "\n".join(summary_lines)
        return summary
    
    async def get_streak_info(self, user_email: str) -> str:
        from app.services.ai_services.goals_service import get_goals_service
        try:
            goals_service = get_goals_service()
            goal_stats = await goals_service.get_goal_stats(user_email)
            weekly_streak = goal_stats.weekly_streak if goal_stats else 0
            total_weekly_streak_count = goal_stats.total_weekly_streak_count if goal_stats else 0
            
            if weekly_streak == 0:
                return f"Current streak: 0 days. Total weeks completed: {total_weekly_streak_count}."
            elif weekly_streak == 1:
                return f"Current streak: {weekly_streak} day! Total weeks completed: {total_weekly_streak_count}."
            else:
                return f"Current streak: {weekly_streak} days! Total weeks completed: {total_weekly_streak_count}."
        except Exception as e:
            logger.warning(f"Failed to get streak info for {user_email}: {str(e)}")
            return "Streak information unavailable."

    async def get_today_goals_completion_stats(self, user_email: str, cutoff_time: Optional[datetime] = None) -> Dict[str, Any]:
        from zoneinfo import ZoneInfo
       
        user = await self.users_collection.find_one({"email": user_email})
        if not user or "timezone" not in user:
            return {
                'total_scheduled': 0,
                'completed': 0,
                'incomplete': 0,
                'completion_percentage': 0.0,
                'is_majority_incomplete': False,
                'is_majority_complete': False,
                'missed_tasks': []
            }
       
        tz_name = str(user["timezone"]).replace(" ", "_")
        try:
            user_tz = ZoneInfo(tz_name)
        except Exception:
            user_tz = ZoneInfo("UTC")
       
        now_user_tz = datetime.now(user_tz)
        today_start = now_user_tz.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = now_user_tz.replace(hour=23, minute=59, second=59, microsecond=999999)
       
        if cutoff_time:
            if cutoff_time.tzinfo is None:
                cutoff_time = cutoff_time.replace(tzinfo=ZoneInfo("UTC"))
            cutoff_time_user_tz = cutoff_time.astimezone(user_tz)
        else:
            cutoff_time_user_tz = None
       
        goals = await self.goals_collection.find({
            "user_email": user_email,
            "completed": {"$ne": True}
        }).to_list(None)
       
        if not goals:
            return {
                'total_scheduled': 0,
                'completed': 0,
                'incomplete': 0,
                'completion_percentage': 0.0,
                'is_majority_incomplete': False,
                'is_majority_complete': False,
                'missed_tasks': []
            }
       
        goal_ids = [str(goal["_id"]) for goal in goals]
        action_items = await self.action_items_collection.find({
            "user_email": user_email,
            "goal_id": {"$in": goal_ids}
        }).to_list(None)
       
        action_items = self.encryption_service.decrypt_documents_bulk(action_items, ActionItem)
       
        total_scheduled = 0
        completed = 0
        missed_tasks = []
       
        for item in action_items:
            weekly_schedule = item.get("weekly_schedule", {})
            if not weekly_schedule:
                continue
           
            weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            today_weekday = weekday_names[now_user_tz.weekday()]
           
            daily_schedule = weekly_schedule.get(today_weekday)
            if not daily_schedule:
                continue
           
            date_value = daily_schedule.get("date")
            if not date_value:
                continue
           
            if isinstance(date_value, str):
                try:
                    action_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    if date_parser:
                        try:
                            action_date = date_parser.parse(date_value)
                        except ValueError:
                            continue
                    else:
                        continue
            elif isinstance(date_value, datetime):
                action_date = date_value
            else:
                continue
           
            if action_date.tzinfo is None:
                action_date = action_date.replace(tzinfo=ZoneInfo("UTC"))
            action_date_user_tz = action_date.astimezone(user_tz)
           
            if not (today_start.date() <= action_date_user_tz.date() <= today_end.date()):
                continue
           
            start_time_value = daily_schedule.get("start_time")
            end_time_value = daily_schedule.get("end_time")
           
            task_start_time = None
            task_end_time = None
           
            if start_time_value:
                if isinstance(start_time_value, datetime):
                    task_start_time = start_time_value
                    if task_start_time.tzinfo is None:
                        task_start_time = task_start_time.replace(tzinfo=ZoneInfo("UTC"))
                    task_start_time = task_start_time.astimezone(user_tz)
                elif isinstance(start_time_value, str):
                    try:
                        from datetime import time as dt_time
                        time_parts = start_time_value.split(':')
                        if len(time_parts) >= 2:
                            hour = int(time_parts[0])
                            minute = int(time_parts[1])
                            second = int(time_parts[2]) if len(time_parts) > 2 else 0
                            start_time_obj = dt_time(hour, minute, second)
                            task_start_time = datetime.combine(action_date_user_tz.date(), start_time_obj)
                            if action_date_user_tz.tzinfo:
                                task_start_time = task_start_time.replace(tzinfo=action_date_user_tz.tzinfo)
                    except (ValueError, AttributeError):
                        pass
           
            if end_time_value:
                if isinstance(end_time_value, datetime):
                    task_end_time = end_time_value
                    if task_end_time.tzinfo is None:
                        task_end_time = task_end_time.replace(tzinfo=ZoneInfo("UTC"))
                    task_end_time = task_end_time.astimezone(user_tz)
                elif isinstance(end_time_value, str):
                    try:
                        from datetime import time as dt_time
                        time_parts = end_time_value.split(':')
                        if len(time_parts) >= 2:
                            hour = int(time_parts[0])
                            minute = int(time_parts[1])
                            second = int(time_parts[2]) if len(time_parts) > 2 else 0
                            end_time_obj = dt_time(hour, minute, second)
                            task_end_time = datetime.combine(action_date_user_tz.date(), end_time_obj)
                            if action_date_user_tz.tzinfo:
                                task_end_time = task_end_time.replace(tzinfo=action_date_user_tz.tzinfo)
                    except (ValueError, AttributeError):
                        pass
           
            if cutoff_time_user_tz:
                if task_end_time:
                    if task_end_time > cutoff_time_user_tz:
                        if task_start_time and task_start_time <= cutoff_time_user_tz:
                            pass
                        else:
                            continue
                elif task_start_time:
                    if task_start_time > cutoff_time_user_tz:
                        continue
                else:
                    if action_date_user_tz > cutoff_time_user_tz:
                        continue
           
            total_scheduled += 1
            is_completed = daily_schedule.get("complete", False)
           
            if is_completed:
                completed += 1
            else:
                task_title = item.get("title", "Untitled Task")
                missed_tasks.append(task_title)
       
        incomplete = total_scheduled - completed
        completion_percentage = (completed / total_scheduled * 100) if total_scheduled > 0 else 0.0
       
        return {
            'total_scheduled': total_scheduled,
            'completed': completed,
            'incomplete': incomplete,
            'completion_percentage': completion_percentage,
            'is_majority_incomplete': incomplete > (total_scheduled * 0.7) if total_scheduled > 0 else False,
            'is_majority_complete': completion_percentage >= 70.0,
            'missed_tasks': missed_tasks
        }

    async def get_weekly_goals_progress(self, user_email: str) -> Dict[str, Any]:
        from zoneinfo import ZoneInfo
       
        user = await self.users_collection.find_one({"email": user_email})
        if not user or "timezone" not in user:
            return {
                'total_scheduled': 0,
                'completed': 0,
                'completion_percentage': 0.0,
                'day_of_week': 0
            }
       
        tz_name = str(user["timezone"]).replace(" ", "_")
        try:
            user_tz = ZoneInfo(tz_name)
        except Exception:
            user_tz = ZoneInfo("UTC")
       
        now_user_tz = datetime.now(user_tz)
        day_of_week = now_user_tz.weekday()
        days_since_monday = day_of_week
        week_start = (now_user_tz - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = (week_start + timedelta(days=6)).replace(hour=23, minute=59, second=59, microsecond=999999)
       
        goals = await self.goals_collection.find({
            "user_email": user_email,
            "completed": {"$ne": True}
        }).to_list(None)
       
        if not goals:
            return {
                'total_scheduled': 0,
                'completed': 0,
                'completion_percentage': 0.0,
                'day_of_week': day_of_week
            }
       
        goal_ids = [str(goal["_id"]) for goal in goals]
        action_items = await self.action_items_collection.find({
            "user_email": user_email,
            "goal_id": {"$in": goal_ids}
        }).to_list(None)
       
        action_items = self.encryption_service.decrypt_documents_bulk(action_items, ActionItem)
       
        total_scheduled = 0
        completed = 0
       
        weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
       
        for item in action_items:
            weekly_schedule = item.get("weekly_schedule", {})
            if not weekly_schedule:
                continue
           
            for day_name in weekday_names:
                daily_schedule = weekly_schedule.get(day_name)
                if not daily_schedule:
                    continue
               
                date_value = daily_schedule.get("date")
                if not date_value:
                    continue
               
                if isinstance(date_value, str):
                    try:
                        action_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        if date_parser:
                            try:
                                action_date = date_parser.parse(date_value)
                            except ValueError:
                                continue
                        else:
                            continue
                elif isinstance(date_value, datetime):
                    action_date = date_value
                else:
                    continue
               
                if action_date.tzinfo is None:
                    action_date = action_date.replace(tzinfo=ZoneInfo("UTC"))
                action_date_user_tz = action_date.astimezone(user_tz)
               
                if week_start <= action_date_user_tz <= week_end:
                    total_scheduled += 1
                    if daily_schedule.get("complete", False):
                        completed += 1
       
        completion_percentage = (completed / total_scheduled * 100) if total_scheduled > 0 else 0.0
       
        return {
            'week_start': week_start,
            'week_end': week_end,
            'total_scheduled': total_scheduled,
            'completed': completed,
            'completion_percentage': completion_percentage,
            'day_of_week': day_of_week
        }

    async def send_morning_notification(self, email: str, execution_id: str = None):
        import uuid
        if execution_id is None:
            execution_id = str(uuid.uuid4())[:8]
        
        if not await self.is_active_user_on_device(email):
            logger.info(f"[NOTIFICATION] Skipping morning notification for {email}: Not the active account on this device.")
            return

        user = await self.users_collection.find_one({"email": email})
        if not user or not user.get("fcm_token"):
            raise TokenNotFoundError(f"No FCM token registered for {email}")
        fcm_token = user["fcm_token"]

        if not await self._try_claim_notification_slot(fcm_token, "morning", email_debug=email):
            raise NotificationError(f"Morning notification already sent to device for {email} (or device locked)")
       
        chat_summaries = "..."
        
        if not user or "timezone" not in user:
            raise UserNotFoundError(f"User not found or timezone not set: {email}")
       
        from zoneinfo import ZoneInfo
        tz_name = str(user["timezone"]).replace(" ", "_")
        try:
            user_tz = ZoneInfo(tz_name)
        except Exception:
            user_tz = ZoneInfo("UTC")
       
        now_user_tz = datetime.now(user_tz)
        cutoff_time = now_user_tz.replace(hour=7, minute=0, second=0, microsecond=0)
        
        plan_history, recent_notifications, today_stats, weekly_stats, streak_info = await asyncio.gather(
            self.get_goals_summary(email),
            asyncio.to_thread(self.get_recent_notification_jobs, email=email, limit=10),
            self.get_today_goals_completion_stats(email, cutoff_time=cutoff_time),
            self.get_weekly_goals_progress(email),
            self.get_streak_info(email)
        )
        
        completion_context = ""
        if today_stats['total_scheduled'] > 0:
            completion_context = f"You have {today_stats['total_scheduled']} goal(s) scheduled for today. {today_stats['completed']} already completed."
        else:
            completion_context = "No specific goals scheduled for today, but keep working toward your health goals!"
        
        weekly_context = ""
        if weekly_stats['total_scheduled'] > 0:
            weekly_context = f" This week: {weekly_stats['completed']}/{weekly_stats['total_scheduled']} goals completed ({weekly_stats['completion_percentage']:.1f}%)."
        
        input_vars = {
            "chat_summaries": chat_summaries,
            "plan_history": plan_history,
            "recent_notifications": recent_notifications,
            "completion_context": completion_context,
            "weekly_context": weekly_context,
            "streak_info": streak_info
        }
        
        schema = {
            "title": "GenerateMorningNotification",
            "description": "Generate a personalized morning notification for the user.",
            "type": "object",
            "properties": {
                "notification": {"type": "string"}
            },
            "required": ["notification"]
        }
        
        system_prompt = "You are a health assistant that sends personalized morning notifications to users."
        user_prompt = GENERATE_MORNING_NOTIFICATION_PROMPT
        
        llm_result = await self._invoke_structured_llm(
            schema=schema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_vars=input_vars
        )
        
        title = "Good Morning!"
        body = llm_result.get("notification", "Start your day with positivity!")
        await self.send_fcm_notification(email, title, body, execution_id=execution_id)

    async def send_evening_notification(self, email: str, execution_id: str = None):
        import uuid
        if execution_id is None:
            execution_id = str(uuid.uuid4())[:8]
       
        if not await self.is_active_user_on_device(email):
            logger.info(f"[NOTIFICATION] Skipping evening notification for {email}: Not the active account on this device.")
            return

        user = await self.users_collection.find_one({"email": email})
        if not user or not user.get("fcm_token"):
            raise TokenNotFoundError(f"No FCM token registered for {email}")
        fcm_token = user["fcm_token"]

        if not await self._try_claim_notification_slot(fcm_token, "evening", email_debug=email):
            raise NotificationError(f"Evening notification already sent to device for {email} (or device locked)")
       
        chat_summaries = "..."
       
        if not user or "timezone" not in user:
            raise UserNotFoundError(f"User not found or timezone not set: {email}")
       
        from zoneinfo import ZoneInfo
        tz_name = str(user["timezone"]).replace(" ", "_")
        try:
            user_tz = ZoneInfo(tz_name)
        except Exception:
            user_tz = ZoneInfo("UTC")
       
        now_user_tz = datetime.now(user_tz)
        cutoff_time = now_user_tz.replace(hour=17, minute=30, second=0, microsecond=0)
       
        plan_history, recent_notifications, today_stats, weekly_stats = await asyncio.gather(
            self.get_goals_summary(email),
            asyncio.to_thread(self.get_recent_notification_jobs, email=email, limit=10),
            self.get_today_goals_completion_stats(email, cutoff_time=cutoff_time),
            self.get_weekly_goals_progress(email)
        )
        
        completion_context = ""
        if today_stats['total_scheduled'] > 0:
            if today_stats['is_majority_incomplete']:
                missed_tasks = today_stats.get('missed_tasks', [])
                if missed_tasks:
                    missed_tasks_str = ", ".join(missed_tasks[:5])
                    if len(missed_tasks) > 5:
                        missed_tasks_str += f" and {len(missed_tasks) - 5} more"
                    completion_context = f"You missed the majority of your goals today ({today_stats['completed']}/{today_stats['total_scheduled']} completed). Missed tasks: {missed_tasks_str}. Politely encourage them to do better tomorrow."
                else:
                    completion_context = f"You missed the majority of your goals today ({today_stats['completed']}/{today_stats['total_scheduled']} completed). Politely encourage them to do better tomorrow."
            else:
                completion_context = f"Congratulations! You completed the majority of your assigned goals today ({today_stats['completed']}/{today_stats['total_scheduled']} completed). Give a positive, encouraging message."
        else:
            completion_context = "No goals scheduled for today."
        
        weekly_context = ""
        if weekly_stats['day_of_week'] > 0:
            weekly_context = f" Weekly progress: {weekly_stats['completed']}/{weekly_stats['total_scheduled']} goals completed this week ({weekly_stats['completion_percentage']:.1f}%)."
        
        streak_info = await self.get_streak_info(email)
        
        input_vars = {
            "chat_summaries": chat_summaries,
            "plan_history": plan_history,
            "recent_notifications": recent_notifications,
            "completion_context": completion_context,
            "weekly_context": weekly_context,
            "streak_info": streak_info
        }
        
        schema = {
            "title": "GenerateEveningNotification",
            "description": "Generate a personalized evening notification for the user.",
            "type": "object",
            "properties": {
                "notification": {"type": "string"}
            },
            "required": ["notification"]
        }
        
        system_prompt = "You are a health assistant that sends personalized evening notifications to users. Keep messages short, sweet, and encouraging like Google Fit or Samsung Fitbit notifications."
        user_prompt = GENERATE_EVENING_NOTIFICATION_PROMPT
        
        llm_result = await self._invoke_structured_llm(
            schema=schema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_vars=input_vars
        )
        
        title = "Good Evening!"
        body = llm_result.get("notification", "Reflect on your progress today!")
        await self.send_fcm_notification(email, title, body, execution_id=execution_id)

    async def send_night_notification(self, email: str, execution_id: str = None):
        import uuid
        if execution_id is None:
            execution_id = str(uuid.uuid4())[:8]
       
        if not await self.is_active_user_on_device(email):
            logger.info(f"[NOTIFICATION] Skipping night notification for {email}: Not the active account on this device.")
            return

        user = await self.users_collection.find_one({"email": email})
        if not user or not user.get("fcm_token"):
            raise TokenNotFoundError(f"No FCM token registered for {email}")
        fcm_token = user["fcm_token"]

        if not await self._try_claim_notification_slot(fcm_token, "night", email_debug=email):
            raise NotificationError(f"Night notification already sent to device for {email} (or device locked)")
       
        chat_summaries = "..."
       
        if not user or "timezone" not in user:
            raise UserNotFoundError(f"User not found or timezone not set: {email}")
       
        from zoneinfo import ZoneInfo
        tz_name = str(user["timezone"]).replace(" ", "_")
        try:
            user_tz = ZoneInfo(tz_name)
        except Exception:
            user_tz = ZoneInfo("UTC")
       
        now_user_tz = datetime.now(user_tz)
        cutoff_time = now_user_tz.replace(hour=20, minute=30, second=0, microsecond=0)
       
        plan_history, recent_notifications, today_stats, weekly_stats = await asyncio.gather(
            self.get_goals_summary(email),
            asyncio.to_thread(self.get_recent_notification_jobs, email=email, limit=10),
            self.get_today_goals_completion_stats(email, cutoff_time=cutoff_time),
            self.get_weekly_goals_progress(email)
        )
        
        completion_context = ""
        if today_stats['total_scheduled'] > 0:
            if not today_stats['is_majority_complete']:
                missed_tasks = today_stats.get('missed_tasks', [])
                if missed_tasks:
                    missed_tasks_str = ", ".join(missed_tasks[:5])
                    if len(missed_tasks) > 5:
                        missed_tasks_str += f" and {len(missed_tasks) - 5} more"
                    completion_context = f"Less than 70% of today's goals were completed ({today_stats['completed']}/{today_stats['total_scheduled']}). Missed tasks: {missed_tasks_str}. Give motivation to complete tomorrow's goals."
                else:
                    completion_context = f"Less than 70% of today's goals were completed ({today_stats['completed']}/{today_stats['total_scheduled']}). Give motivation to complete tomorrow's goals."
            else:
                completion_context = f"Great job! You completed {today_stats['completed']}/{today_stats['total_scheduled']} goals today. Give a positive good night message and best of luck for tomorrow."
        else:
            completion_context = "No goals scheduled for today."
        
        weekly_context = ""
        if weekly_stats['day_of_week'] > 0:
            weekly_context = f" Weekly progress: {weekly_stats['completed']}/{weekly_stats['total_scheduled']} goals completed this week ({weekly_stats['completion_percentage']:.1f}%)."
        
        streak_info = await self.get_streak_info(email)
        
        input_vars = {
            "todays_timestamp": datetime.now().isoformat(),
            "chat_summaries": chat_summaries,
            "plan_history": plan_history,
            "recent_notifications": recent_notifications,
            "completion_context": completion_context,
            "weekly_context": weekly_context,
            "streak_info": streak_info
        }
        
        schema = {
            "title": "GenerateNightNotification",
            "description": "Generate a personalized night notification for the user.",
            "type": "object",
            "properties": {
                "notification": {"type": "string"}
            },
            "required": ["notification"]
        }
        
        system_prompt = "You are a health assistant that sends personalized night notifications to users. Keep messages short, sweet, and encouraging like Google Fit or Samsung Fitbit notifications."
        user_prompt = GENERATE_NIGHT_NOTIFICATION_PROMPT
        
        llm_result = await self._invoke_structured_llm(
            schema=schema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_vars=input_vars
        )
        
        title = "Good Night!"
        body = llm_result.get("notification", "Rest well and recharge for tomorrow!")
        await self.send_fcm_notification(email, title, body, execution_id=execution_id)

    async def send_checkin_notification(self, email: str, execution_id: str = None):
        import uuid
        if execution_id is None:
            execution_id = str(uuid.uuid4())[:8]
       
        if not await self.is_active_user_on_device(email):
            logger.info(f"[NOTIFICATION] Skipping checkin notification for {email}: Not the active account on this device.")
            return

        user = await self.users_collection.find_one({"email": email})
        if not user or not user.get("fcm_token"):
            raise TokenNotFoundError(f"No FCM token registered for {email}")
        fcm_token = user["fcm_token"]

        if not await self._try_claim_notification_slot(fcm_token, "checkin", email_debug=email):
            raise NotificationError(f"Check-in notification already sent to device for {email} (or device locked)")
       
        chat_summaries = "..."
       
        if not user or "timezone" not in user:
            raise UserNotFoundError(f"User not found or timezone not set: {email}")
       
        from zoneinfo import ZoneInfo
        tz_name = str(user["timezone"]).replace(" ", "_")
        try:
            user_tz = ZoneInfo(tz_name)
        except Exception:
            user_tz = ZoneInfo("UTC")
       
        now_user_tz = datetime.now(user_tz)
        cutoff_time = now_user_tz.replace(hour=12, minute=0, second=0, microsecond=0)
       
        plan_history, recent_notifications, today_stats, weekly_stats, streak_info = await asyncio.gather(
            self.get_goals_summary(email),
            asyncio.to_thread(self.get_recent_notification_jobs, email=email, limit=10),
            self.get_today_goals_completion_stats(email, cutoff_time=cutoff_time),
            self.get_weekly_goals_progress(email),
            self.get_streak_info(email)
        )
        
        completion_context = ""
        if today_stats['total_scheduled'] > 0:
            if today_stats['is_majority_complete']:
                completion_context = f"Great progress! You've completed {today_stats['completed']}/{today_stats['total_scheduled']} goals so far today ({today_stats['completion_percentage']:.1f}%). Keep it up!"
            else:
                missed_tasks = today_stats.get('missed_tasks', [])
                if missed_tasks:
                    missed_tasks_str = ", ".join(missed_tasks[:3])
                    if len(missed_tasks) > 3:
                        missed_tasks_str += f" and {len(missed_tasks) - 3} more"
                    completion_context = f"You've completed {today_stats['completed']}/{today_stats['total_scheduled']} goals so far. Still pending: {missed_tasks_str}."
                else:
                    completion_context = f"You've completed {today_stats['completed']}/{today_stats['total_scheduled']} goals so far. Keep pushing toward your goals!"
        else:
            completion_context = "No specific goals scheduled for today, but remember to stay active and healthy!"
        
        weekly_context = ""
        if weekly_stats['total_scheduled'] > 0:
            weekly_context = f" Weekly progress: {weekly_stats['completed']}/{weekly_stats['total_scheduled']} goals completed this week ({weekly_stats['completion_percentage']:.1f}%)."
       
        input_vars = {
            "chat_summaries": chat_summaries,
            "plan_history": plan_history,
            "recent_notifications": recent_notifications,
            "completion_context": completion_context,
            "weekly_context": weekly_context,
            "streak_info": streak_info,
        }
        
        schema = {
            "title": "GenerateCheckinNotification",
            "description": "Generate a personalized check-in notification for the user.",
            "type": "object",
            "properties": {
                "notification": {"type": "string"}
            },
            "required": ["notification"]
        }
        
        system_prompt = "You are a health assistant that sends personalized check-in notifications to users."
        user_prompt = GENERATE_CHECKIN_NOTIFICATION_PROMPT
        
        llm_result = await self._invoke_structured_llm(
            schema=schema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_vars=input_vars
        )
        
        title = "Check-In Time!"
        body = llm_result.get("notification", "How are you progressing with your goals today?")
        await self.send_fcm_notification(email, title, body, execution_id=execution_id)

    async def check_daily_notifications_scheduled(self, email: str) -> bool:
        if self.scheduler is None:
            self._initialize_scheduler()
       
        if not self.scheduler.running:
            await asyncio.sleep(0.5)
        else:
            await asyncio.sleep(0.2)
       
        jobs = self.scheduler.get_jobs()
        nudge_ids = {
            f"morning_{email}",
            f"evening_{email}",
            f"night_{email}",
            f"checkin_{email}",
        }
        existing_ids = {job.id for job in jobs}
        return nudge_ids.issubset(existing_ids)

    async def schedule_daily_notifications(self):
        async with self._scheduling_lock:
            users = await self.users_collection.find({"notifications_enabled": True}).to_list(length=None)
            for user in users:
                email = user["email"]
               
                if self.scheduler is None:
                    self._initialize_scheduler()
               
                if not self.scheduler.running:
                    self.scheduler.start()
                    await asyncio.sleep(1)
               
                if await self.check_daily_notifications_scheduled(email):
                    logger.debug(f"Notifications already scheduled for {email}, skipping")
                    continue
                    
                if "timezone" not in user or not user.get("timezone"):
                    logger.warning(f"Skipping daily notification scheduling for {email}: timezone not set")
                    continue
                    
                try:
                    local_dt = datetime.now().replace(hour=7, minute=0, second=0, microsecond=0)
                    run_dt_utc = await self.local_time_to_utc(local_dt, email)
                    MORNING_HOUR = run_dt_utc.hour
                    MORNING_MINUTE = run_dt_utc.minute
                    
                    local_dt = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
                    run_dt_utc = await self.local_time_to_utc(local_dt, email)
                    CHECKIN_HOUR = run_dt_utc.hour
                    CHECKIN_MINUTE = run_dt_utc.minute
                    
                    local_dt = datetime.now().replace(hour=17, minute=30, second=0, microsecond=0)
                    run_dt_utc = await self.local_time_to_utc(local_dt, email)
                    EVENING_HOUR = run_dt_utc.hour
                    EVENING_MINUTE = run_dt_utc.minute
                    
                    local_dt = datetime.now().replace(hour=20, minute=30, second=0, microsecond=0)
                    run_dt_utc = await self.local_time_to_utc(local_dt, email)
                    NIGHT_HOUR = run_dt_utc.hour
                    NIGHT_MINUTE = run_dt_utc.minute
                    
                    self.scheduler.add_job(
                        morning_notification_job,
                        args=[email],
                        trigger='cron',
                        hour=MORNING_HOUR,
                        minute=MORNING_MINUTE,
                        id=f"morning_{email}",
                        replace_existing=True,
                        coalesce=True,
                        misfire_grace_time=15 * 60,
                    )
                    self.scheduler.add_job(
                        evening_notification_job,
                        args=[email],
                        trigger='cron',
                        hour=EVENING_HOUR,
                        minute=EVENING_MINUTE,
                        id=f"evening_{email}",
                        replace_existing=True,
                        coalesce=True,
                        misfire_grace_time=15 * 60,
                    )
                    self.scheduler.add_job(
                        night_notification_job,
                        args=[email],
                        trigger='cron',
                        hour=NIGHT_HOUR,
                        minute=NIGHT_MINUTE,
                        id=f"night_{email}",
                        replace_existing=True,
                        coalesce=True,
                        misfire_grace_time=15 * 60,
                    )
                    self.scheduler.add_job(
                        checkin_notification_job,
                        args=[email],
                        trigger='cron',
                        hour=CHECKIN_HOUR,
                        minute=CHECKIN_MINUTE,
                        id=f"checkin_{email}",
                        replace_existing=True,
                        coalesce=True,
                        misfire_grace_time=15 * 60,
                    )
                    await self.users_collection.update_one({"email": email}, {"$set": {"daily_notifications": True}})
                except UserNotFoundError as e:
                    logger.warning(f"Skipping daily notification scheduling for {email}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error scheduling daily notifications for {email}: {str(e)}", exc_info=True)
       
    def get_recent_notification_jobs(self, email: str, limit: int = 10) -> List[Dict]:
        if self.scheduler is None:
            self._initialize_scheduler()
        jobs = self.scheduler.get_jobs()
        user_jobs = [
            job for job in jobs
            if job.args and len(job.args) > 0 and job.args[0] == email
        ]
        user_jobs.sort(key=lambda job: job.next_run_time or datetime.min, reverse=True)
        return [
            {
                "id": job.id,
                "next_run_time": job.next_run_time,
                "func": str(job.func),
                "args": job.args,
                "kwargs": job.kwargs,
            }
            for job in user_jobs[:limit]
        ]

_nudge_service_instance = None

def get_nudge_service():
    global _nudge_service_instance
    if _nudge_service_instance is None:
        _nudge_service_instance = NudgeService()
    return _nudge_service_instance
