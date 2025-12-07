import base64
import io
import logging
from datetime import datetime
from typing import Tuple

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status, Depends
from pydantic import EmailStr

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from ...config import ALLOWED_PROFILE_IMAGE_MIME_TYPES, PROFILE_PICTURE_MAX_BYTES
from ...schemas.backend.profile import UserProfileUpdate
from ...schemas.backend.user import UserPreferences
from ...services.backend_services.db import get_db
from app.core.security import get_current_user, CurrentUser

logger = logging.getLogger(__name__)
user_router = APIRouter()
db = get_db()

MAX_PROFILE_IMAGE_SIZE = (1024, 1024)  
PROFILE_IMAGE_QUALITY = 85  

@user_router.get("/api/user/preferences")
async def get_user_preferences(current_user: CurrentUser = Depends(get_current_user)):
    prefs = await db["preferences"].find_one({"email": current_user.email})
    return {"exists": bool(prefs)}

@user_router.post("/api/user/preferences")
async def save_user_preferences(
    prefs: UserPreferences,
    current_user: CurrentUser = Depends(get_current_user)
):
    preferences = db["preferences"]
    # Override email from token
    prefs_dict = prefs.dict()
    prefs_dict["email"] = current_user.email
    await preferences.update_one(
        {"email": current_user.email},
        {"$set": prefs_dict},
        upsert=True
    )
    return {"message": "Preferences saved"}

@user_router.get("/api/user/profile")
async def get_user_profile(current_user: CurrentUser = Depends(get_current_user)):
    """Get user profile information."""
    profile = await db["users"].find_one({"email": current_user.email}, {"_id": 0})
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Ensure all required fields exist in the response
    defaults = {
        "name": "",
        "phone_number": "",
        "date_of_birth": "",
        "blood_type": "",
        "notifications_enabled": True,
        "profile_picture": None,
        "timezone": None
    }
    profile = {**defaults, **profile}  # Merge defaults with actual data
    
    return {
        "email": profile["email"],
        "name": profile["name"],
        "phone_number": profile["phone_number"],
        "date_of_birth": profile["date_of_birth"],
        "blood_type": profile["blood_type"],
        "notifications_enabled": profile["notifications_enabled"],
        "profile_picture": profile.get("profile_picture"),
        "timezone": profile.get("timezone")
    }

@user_router.post("/api/user/profile/update")
async def update_user_profile(
    profile: UserProfileUpdate,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Update user profile information."""
    users = db["users"]
    
    # First check if user exists
    existing_user = await users.find_one({"email": current_user.email})
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # Start with only tracking the update time
        update_data = {
            "updated_at": datetime.utcnow()  # Track when profile was last updated
        }

        # Only include fields that were sent in the request and are not None/empty
        if hasattr(profile, 'full_name') and profile.full_name is not None and profile.full_name.strip():
            update_data["name"] = profile.full_name
            
        # Only update these fields if they were explicitly provided (not None)
        for field, db_field in [
            ('phone_number', 'phone_number'),
            ('date_of_birth', 'date_of_birth'),
            ('blood_type', 'blood_type'),
            ('timezone', 'timezone')
        ]:
            if hasattr(profile, field) and getattr(profile, field) is not None:
                value = getattr(profile, field)
                # Skip empty strings
                if isinstance(value, str) and not value.strip():
                    continue
                # Special handling for blood type
                if field == 'blood_type' and value:
                    value = value.upper()
                update_data[db_field] = value
        
        # Handle notifications_enabled separately as it's a boolean
        if hasattr(profile, 'notifications_enabled') and profile.notifications_enabled is not None:
            update_data['notifications_enabled'] = profile.notifications_enabled
        
        # Update user profile and return updated document in one operation using find_one_and_update
        # This eliminates the need for a separate find_one query after update
        updated_profile = await users.find_one_and_update(
            {"email": current_user.email},
            {"$set": update_data},
            return_document=True,
            projection={"_id": 0}  # Exclude _id from response
        )
        
        if not updated_profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        defaults = {
            "name": "",
            "phone_number": "",
            "date_of_birth": "",
            "blood_type": "",
            "notifications_enabled": True,
            "profile_picture": None,
            "timezone": None
        }
        updated_profile = {**defaults, **updated_profile}
        
        return {
            "message": "Profile updated successfully",
            "profile": {
                "email": updated_profile["email"],
                "name": updated_profile["name"],
                "phone_number": updated_profile["phone_number"],
                "date_of_birth": updated_profile["date_of_birth"],
                "blood_type": updated_profile["blood_type"],
                "notifications_enabled": updated_profile["notifications_enabled"],
                "profile_picture": updated_profile.get("profile_picture"),
                "timezone": updated_profile.get("timezone")
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _encode_data_url(mime_type: str, content: bytes) -> str:
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _compress_image(content: bytes, mime_type: str) -> Tuple[bytes, str]:
    """Compress and resize image to reduce storage size."""
    if not PIL_AVAILABLE:
        return content, mime_type
    
    try:
        image = Image.open(io.BytesIO(content))
        
        # Convert RGBA to RGB if needed (for JPEG compatibility)
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if larger than max size
        if image.size[0] > MAX_PROFILE_IMAGE_SIZE[0] or image.size[1] > MAX_PROFILE_IMAGE_SIZE[1]:
            image.thumbnail(MAX_PROFILE_IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Compress to JPEG bytes
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=PROFILE_IMAGE_QUALITY, optimize=True)
        return output.getvalue(), 'image/jpeg'
    except Exception as e:
        logger.error(f"[Upload] Image compression failed: {e}")
        return content, mime_type


async def _prepare_upload(upload: UploadFile) -> Tuple[str, int]:
    content = await upload.read()
    
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    original_size = len(content)
    if original_size > PROFILE_PICTURE_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Profile picture exceeds the 10MB limit.",
        )

    mime_type = (upload.content_type or "").lower()
    if mime_type not in ALLOWED_PROFILE_IMAGE_MIME_TYPES:
        allowed = ", ".join(sorted(ALLOWED_PROFILE_IMAGE_MIME_TYPES))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type. Allowed types: {allowed}",
        )

    compressed_content, final_mime_type = _compress_image(content, mime_type)
    final_size = len(compressed_content)
    data_url = _encode_data_url(final_mime_type, compressed_content)
    
    return data_url, final_size


@user_router.post("/api/user/profile/upload-picture")
async def upload_profile_picture(
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Upload a profile picture for a user."""
    users = db["users"]
    existing_user = await users.find_one({"email": current_user.email})
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        data_url, size = await _prepare_upload(file)

        result = await users.update_one(
            {"email": current_user.email},
            {
                "$set": {
                    "profile_picture": data_url,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
     
        if result.matched_count == 0:
            raise HTTPException(status_code=500, detail="Failed to update profile picture")
        
        return {
            "message": "Profile picture updated successfully",
            "profile_picture": data_url,
            "size_bytes": size
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[Upload] Error during upload for {current_user.email}: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")
