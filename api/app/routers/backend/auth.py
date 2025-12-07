# TODO: Auth Service is not implemented yet here

import asyncio
from datetime import datetime, timezone
from zoneinfo import available_timezones

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...schemas.backend.user import OTPVerify, UserLogin, UserRegister
from ...services.backend_services.db import get_db
from ...services.backend_services.email_utils import generate_otp, send_email, is_demo_account, DEMO_ACCOUNT_OTP
from app.core.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_bearer_token,
    get_current_user,
    CurrentUser,
)
from app.services.backend_services.token_service import (
    get_refresh_token_service,
)

auth_router = APIRouter()
db = get_db()
refresh_token_service = get_refresh_token_service()


class RefreshTokenRequest(BaseModel):
    refresh_token: str

@auth_router.post("/register")
async def register(user: UserRegister):
    users = db["users"]
    existing_user = await users.find_one({"email": user.email})
    if existing_user and existing_user.get("verified", True):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists."
        )
    # Temporarily disabled: All users use OTP "123456", no email sending
    otp = "123456"
    user_dict = {
        "name": user.name,
        "email": user.email,
        "otp": otp,
        "verified": False,
        "daily_notifications": False,
        "timezone": user.timezone,
        "token_version": 0,
    }
    send_otp = False
    if existing_user:
        if not existing_user.get("verified", False):
            await users.update_one({"email": user.email}, {"$set": user_dict})
            send_otp = True
        else:
            # Avoid email enumeration by returning a generic response
            send_otp = False
    else:
        await users.insert_one(user_dict)
        send_otp = True

    # Email sending temporarily disabled
    # if send_otp:
    #     await asyncio.to_thread(
    #         send_email,
    #         user.email,
    #         "Your OTP for Registration",
    #         f"Your OTP is: {otp}"
    #     )
    return {"message": "If an account exists for this email, an OTP has been sent. Please verify to complete registration."}

@auth_router.get("/available-timezones")
async def timezones():
    zones = sorted(available_timezones())
    return {"timezones": zones}

@auth_router.post("/login")
async def login(user: UserLogin):
    users = db["users"]
    db_user = await users.find_one({"email": user.email})
    if not db_user:
        raise HTTPException(status_code=404, detail="User doesn't exist. Please signup!")
    if not db_user.get("verified", False):
        raise HTTPException(status_code=403, detail="User not verified. Please register and verify your email.")
    # Temporarily disabled: All users use OTP "123456", no email sending
    otp = "123456"
    await users.update_one({"email": user.email}, {"$set": {"otp": otp}})
    # Email sending temporarily disabled
    # send_email(user.email, "Your OTP for Login", f"Your OTP is: {otp}")
    return {"message": "OTP sent to email. Please verify to complete login."}

@auth_router.post("/verify-registration-otp")
async def verify_registration_otp(data: OTPVerify):
    users = db["users"]
    db_user = await users.find_one({"email": data.email})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if db_user.get("verified", False):
        # User already verified, issue tokens anyway
        response_payload = await _issue_tokens_response(db_user)
        response_payload["message"] = "User already verified"
        return response_payload
    
    # Temporarily: Accept "123456" for all users, or existing OTP validation
    otp_valid = (data.otp == "123456") or (is_demo_account(data.email) and data.otp == DEMO_ACCOUNT_OTP) or (db_user.get("otp") == data.otp)
    
    if otp_valid:
        token_version = db_user.get("token_version", 0)
        if "token_version" not in db_user:
            token_version = 0
            await users.update_one(
                {"email": data.email},
                {"$set": {"token_version": token_version}},
            )
        await users.update_one(
            {"email": data.email}, 
            {
                "$set": {
                    "verified": True, 
                    "otp": None,
                    "last_active_at": datetime.now(timezone.utc)
                }
            }
        )
        # Fetch updated user data
        db_user = await users.find_one({"email": data.email})
        response_payload = await _issue_tokens_response(db_user)
        response_payload["message"] = "Registration verified successfully"
        return response_payload
    else:
        raise HTTPException(status_code=400, detail="Invalid OTP")

@auth_router.post("/verify-login-otp")
async def verify_login_otp(data: OTPVerify):
    users = db["users"]
    db_user = await users.find_one({"email": data.email})
    if not db_user or not db_user.get("verified", False):
        raise HTTPException(status_code=404, detail="User not found or not verified")
    
    # Temporarily: Accept "123456" for all users, or existing OTP validation
    otp_valid = (data.otp == "123456") or (is_demo_account(data.email) and data.otp == DEMO_ACCOUNT_OTP) or (db_user.get("otp") == data.otp)
    
    if otp_valid:
        token_version = db_user.get("token_version", 0)
        if "token_version" not in db_user:
            token_version = 0
            await users.update_one(
                {"email": data.email},
                {"$set": {"token_version": token_version}},
            )
        await users.update_one(
            {"email": data.email}, 
            {
                "$set": {
                    "otp": None,
                    "last_active_at": datetime.now(timezone.utc)
                }
            }
        )
        response_payload = await _issue_tokens_response(db_user)
        response_payload["message"] = "Login successful"
        return response_payload
    else:
        raise HTTPException(status_code=400, detail="Invalid OTP")


@auth_router.post("/refresh")
async def refresh_access_token(payload: RefreshTokenRequest):
    try:
        token_doc, new_refresh_token, refresh_expires_at = (
            await refresh_token_service.rotate_refresh_token(
                payload.refresh_token
            )
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    users = db["users"]
    db_user = await users.find_one({"email": token_doc["user_email"]})
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with token no longer exists",
        )

    if db_user.get("token_version", 0) != token_doc.get("token_version", 0):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked. Please login again.",
        )

    response_payload = await _issue_tokens_response(
        db_user,
        provided_refresh_token=new_refresh_token,
        provided_refresh_expires_at=refresh_expires_at,
    )
    response_payload["message"] = "Token refreshed successfully"
    return response_payload


@auth_router.post("/logout")
async def logout(current_user: CurrentUser = Depends(get_current_user)):
    users = db["users"]
    db_user = await users.find_one({"email": current_user.email})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    result = await users.update_one(
        {"email": current_user.email},
        {
            "$inc": {"token_version": 1},
            "$set": {"last_active_at": datetime.now(timezone.utc)},
        },
    )

    await refresh_token_service.revoke_all_tokens_for_user(current_user.email)

    return {"message": "Logout successful"}


async def _issue_tokens_response(
    db_user: dict,
    provided_refresh_token: str | None = None,
    provided_refresh_expires_at: datetime | None = None,
) -> dict:
    token_version = db_user.get("token_version", 0)
    access_token = create_access_token(
        {
            "sub": db_user["email"],
            "name": db_user["name"],
            "token_version": token_version,
        }
    )
    if provided_refresh_token and provided_refresh_expires_at:
        refresh_token = provided_refresh_token
        refresh_expires_at = provided_refresh_expires_at
    else:
        refresh_token, refresh_expires_at = await refresh_token_service.create_refresh_token(
            db_user["email"], token_version
        )

    now = datetime.now(timezone.utc)
    refresh_expires_in = max(
        int((refresh_expires_at - now).total_seconds()), 0
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "refresh_expires_in": refresh_expires_in,
        "user": {"name": db_user["name"], "email": db_user["email"]},
    }
