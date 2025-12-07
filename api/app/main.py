import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.formparsers import MultiPartParser
from starlette.requests import Request as StarletteRequest

from app.config import (
    DEBUG_COLLECTION_NAME,
    PROFILE_PICTURE_MAX_BYTES,
    MULTIPART_SPOOL_MAX_SIZE,
    MULTIPART_MAX_FILES,
    MULTIPART_MAX_FIELDS,
)
from app.exceptions import AppException, custom_exception_handler, generic_exception_handler
from app.routers.ai.chat import chat_router
from app.routers.ai.goals import goals_router
from app.routers.ai.lab_report import lab_report_router
from app.routers.ai.personalization_profile import personalization_router
from app.routers.backend.auth import auth_router
from app.routers.backend.health_alert import health_alert_router
from app.routers.backend.health_score import health_score_router
from app.routers.backend.nudge import nudge_router
from app.routers.backend.delete_account import delete_account_router
from app.routers.backend.preferences import preferences_router
from app.routers.backend.upload import upload_router
from app.routers.backend.user import user_router
from app.routers.backend.review import review_router
from app.routers.backend.app_version import app_version_router
from app.services.backend_services.db import close_db, get_client, get_db
from app.services.backend_services.email_utils import send_email
from app.services.backend_services.nudge_service import get_nudge_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scar.api")


def stop_scheduler(scheduler):
    scheduler.shutdown()
    logger.info("🛑 Nudge scheduler stopped")




@asynccontextmanager
async def lifespan(app: FastAPI):
    get_db()
    nudge_service = get_nudge_service()
    nudge_service.start_scheduler()
    print("🚀 Nudge scheduler started")
    # Schedule notifications in background, don't block startup
    task = asyncio.create_task(nudge_service.schedule_daily_notifications())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        nudge_service.stop_scheduler()
        await close_db()
    print("✅ Shutdown complete")


class TenMBMultiPartParser(MultiPartParser):
    spool_max_size = MULTIPART_SPOOL_MAX_SIZE  
    max_part_size = PROFILE_PICTURE_MAX_BYTES  

    def __init__(
        self,
        headers,
        stream,
        *,
        max_files=MULTIPART_MAX_FILES,
        max_fields=MULTIPART_MAX_FIELDS,
        max_part_size=PROFILE_PICTURE_MAX_BYTES,
    ):
        super().__init__(
            headers,
            stream,
            max_files=max_files,
            max_fields=max_fields,
            max_part_size=max_part_size,
        )


class LargeMultipartRequest(StarletteRequest):
    async def _get_form(
        self,
        *,
        max_files: int | float = MULTIPART_MAX_FILES,
        max_fields: int | float = MULTIPART_MAX_FIELDS,
    ):
        return await super()._get_form(
            max_files=max_files,
            max_fields=max_fields,
        )

    def form(
        self,
        *,
        max_files: int | float = MULTIPART_MAX_FILES,
        max_fields: int | float = MULTIPART_MAX_FIELDS,
    ):
        return super().form(
            max_files=max_files,
            max_fields=max_fields,
        )

app = FastAPI(
    title="Medical RAG API", 
    description="Medical RAG API with async support",
    version="1.0.0",
    lifespan=lifespan,
    request_class=LargeMultipartRequest,
)

app.router.default_form_parser_class = TenMBMultiPartParser

app.add_exception_handler(AppException, custom_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add request logging middleware to track all incoming requests (including failed ones)
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    client_ip = request.client.host if request.client else 'unknown'
    has_auth = "Authorization" in request.headers
    logger.info(f"[REQUEST] {request.method} {request.url.path} - Client: {client_ip} - Auth: {has_auth}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"[REQUEST] {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[REQUEST] {request.method} {request.url.path} - ERROR: {str(e)} - Time: {process_time:.3f}s", exc_info=True)
        raise

app.include_router(auth_router, tags=["auth"])
app.include_router(user_router, tags=["user"])
app.include_router(chat_router, tags=["chat"])
app.include_router(upload_router, tags=["upload"])
app.include_router(goals_router, tags=["goals"])
app.include_router(lab_report_router, tags=["lab-reports"])
app.include_router(personalization_router, tags=["personalization"])
app.include_router(delete_account_router, tags=["delete-account"])
app.include_router(preferences_router, tags=["preferences"])
app.include_router(nudge_router, prefix="/api/nudge", tags=["nudge"])
app.include_router(health_alert_router, prefix="/api/health-alert", tags=["health-alert"])
app.include_router(health_score_router, prefix="/api/health-score", tags=["health-score"])
app.include_router(review_router, tags=["review"])
app.include_router(app_version_router, tags=["app-version"])

@app.get("/")
async def root():
    try:
        db = get_db()
        client = get_client()
        
        collections = await db.list_collection_names()
        server_info = await client.server_info()
        
        return {
            "message": "Hello World", 
            "mongodb": "connected", 
            "collections_count": len(collections),
            "mongodb_version": server_info.get("version", "unknown")
        }
    except Exception as e:
        return {"message": "Hello World", "mongodb": f"connection error: {str(e)}"}

@app.post("/send-email")
def send_email_endpoint(
    to_email: str = Body(...),
    subject: str = Body(...),
    body: str = Body(...)
):
    send_email(to_email, subject, body)
    return {"message": "Email sent successfully"}

@app.post("/debug/store")
async def store_debug_data(data: dict = Body(...), email: str = None):
    if DEBUG_COLLECTION_NAME == None:
        return {"error": "DEBUG_COLLECTION_NAME is not set in configuration."}
    try:
        db = get_db()
        debug_collection = db[DEBUG_COLLECTION_NAME]
        debug_entry = {
            "data": data,
            "email": email,
            "timestamp": datetime.now(timezone.utc)
        }
        result = await debug_collection.insert_one(debug_entry)
        return {
            "message": "Debug data stored successfully",
            "id": str(result.inserted_id),
            "email": email,
            "timestamp": debug_entry["timestamp"].isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to store debug data: {str(e)}"}

@app.get("/debug/store")
async def get_debug_data(email: str = None):
    try:
        db = get_db()
        debug_collection = db[DEBUG_COLLECTION_NAME]
        query = {"email": email} if email else {}
        debug_entries = await debug_collection.find(query).sort("timestamp", -1).to_list(length=None)
        
        for entry in debug_entries:
            entry["_id"] = str(entry["_id"])
            entry["timestamp"] = entry["timestamp"].isoformat()
        
        return {
            "message": "Debug data retrieved successfully",
            "count": len(debug_entries),
            "data": debug_entries
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to store debug data")
