from pydantic import BaseModel, Field
from typing import Union, List, Optional
from datetime import datetime
from enum import Enum
from app.schemas.backend.encrypt import EncryptedField


class HealthMetric(BaseModel):
    value: Union[int, float]
    unit: str
    isAvailable: bool
    error: str


class HealthMetricData(BaseModel):
    steps: HealthMetric
    heartRate: HealthMetric
    activeEnergy: HealthMetric
    sleep: HealthMetric
    weight: HealthMetric
    bodyFat: HealthMetric
    bloodGlucose: HealthMetric
    oxygenSaturation: HealthMetric


class HealthMetricHourlyData(BaseModel):
    data: HealthMetricData
    created_at: datetime


class StepSummary(BaseModel):
    total: int


class HeartRateSummary(BaseModel):
    average: float
    min: int  
    max: int


class ActiveEnergySummary(BaseModel):
    total: int


class SleepSummary(BaseModel):
    totalHours: float


class WeightSummary(BaseModel):
    value: float
    unit: str 


class BodyFatSummary(BaseModel):
    value: float
    unit: str 


class BloodGlucoseSummary(BaseModel):
    average: float
    max: float
    min: float


class OxygenSaturationSummary(BaseModel):
    average: float
    max: float
    min: float


class AggregatedHealthSummary(BaseModel):
    date: datetime
    step: StepSummary
    heartRate: HeartRateSummary
    activeEnergy: ActiveEnergySummary
    sleep: SleepSummary
    weight: WeightSummary
    bodyFat: BodyFatSummary
    bloodGlucose: BloodGlucoseSummary
    oxygenSaturation: OxygenSaturationSummary


class StatusEnum(str, Enum):
    active = "active"
    resolved = "resolved"


class SeverityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class MetricEnum(str, Enum):
    steps = "steps"
    heartRate = "heartRate"
    activeEnergy = "activeEnergy"
    sleep = "sleep"
    weight = "weight"
    bodyFat = "bodyFat"
    bloodGlucose = "bloodGlucose"
    oxygenSaturation = "oxygenSaturation"
    dataQuality = "dataQuality"


# NOTE: HealthAlert class is defined later (line 169) as HealthAlert(HealthAlertCreate)
# This duplicate definition was removed to avoid confusion
# The actual HealthAlert class extends HealthAlertCreate and includes all fields

# TODO: Migrate user_email to user_id in future
class HealthDataCreate(BaseModel):
    user_email: str
    created_at: datetime
    hourly_data: List[HealthMetricHourlyData]
    aggregated_summary: Union[AggregatedHealthSummary, None] = None


class HealthData(HealthDataCreate):
    id: str

class HealthDataScoreGenerate(BaseModel):
    score: float = Field(..., ge=0, le=100, description="Health score between 0 and 100")
    reasons: List[str] = Field(..., description="List of reasons impacting the health score")

class HealthDataScore(HealthDataScoreGenerate):
    health_data_id: str

# TODO: Make it read and unread
class AlertStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"


class HealthAlertSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HealthAlertGenerate(BaseModel):
    metric: str
    title: str = EncryptedField(...)
    key_point: str
    message: str = EncryptedField(...)
    severity: HealthAlertSeverity
    # Actual metric values from uploaded health data
    current_value: Optional[Union[int, float]] = None
    threshold_value: Optional[Union[int, float]] = None
    unit: Optional[str] = None


# TODO: Look into shifting created_at to HealthAlert
class HealthAlertCreate(HealthAlertGenerate):
    user_email: str
    health_data_id: str
    status: AlertStatus
    created_at: datetime


class HealthAlert(HealthAlertCreate):
    id: str


class HealthAlertGenerationResponse(BaseModel):
    should_generate_alert: bool = Field(
        description="True if an alert should be generated"
    )
    alerts: List[HealthAlertGenerate] = Field(
        default=[], description="List of alerts to generate"
    )
