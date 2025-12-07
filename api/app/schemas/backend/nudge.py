from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from enum import Enum
from datetime import datetime
from typing import Optional
from enum import Enum
from app.schemas.backend.encrypt import EncryptedField

class NudgeType(str, Enum):
    REMINDER = "reminder"
    HEALTH_ALERT = "health_alert"

class NudgeStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"

# TODO: Maybe creating a NudgeCreate schema would be better here
class Nudge(BaseModel):
    id: Optional[str] = None
    user_email: EmailStr
    scheduled_time: datetime
    title: str = EncryptedField(..., min_length=1, max_length=200)
    body: str = EncryptedField(..., min_length=1, max_length=500)
    status: NudgeStatus = NudgeStatus.PENDING
    created_at: Optional[datetime] = None

class GoalNudge(Nudge):
    goal_id: str
    action_item_title: str
    type: NudgeType = NudgeType.REMINDER

class HealthAlertNudge(Nudge):
    health_data_id: str
    type: NudgeType = NudgeType.HEALTH_ALERT

class FCMTokenRequest(BaseModel):
    fcm_token: str

class FCMTokenResponse(BaseModel):
    success: bool
    message: str

class NotificationType(str, Enum):
    MORNING = "morning"
    CHECKIN = "checkin"
    EVENING = "evening"
    NIGHT = "night"

class SendCheckinRequest(BaseModel):
    notification_type: NotificationType

class SendCheckinResponse(BaseModel):
    success: bool
    message: str
