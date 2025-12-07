from pydantic import BaseModel, EmailStr

class NotificationPreferences(BaseModel):
    email: EmailStr
    push_enabled: bool = True
    health_reminders: bool = True
    appointment_alerts: bool = True
    medication_reminders: bool = True
    goal_updates: bool = True
