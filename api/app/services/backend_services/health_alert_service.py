# TODO:
# - Do a proper error handling for each of the functions

from datetime import datetime, timezone, timedelta
from typing import List, Optional
from app.services.backend_services.db import get_db
from app.schemas.backend.health_alert import (
    HealthAlertCreate,
    HealthAlertGenerate,
    HealthData,
    HealthMetricHourlyData,
    AggregatedHealthSummary,
    HealthDataCreate,
    StepSummary,
    HeartRateSummary,
    ActiveEnergySummary,
    SleepSummary,
    WeightSummary,
    BodyFatSummary,
    BloodGlucoseSummary,
    OxygenSaturationSummary,
    HealthMetricData,
    HealthAlert,
    HealthAlertGenerationResponse,
    HealthAlertSeverity,
    AlertStatus,
    HealthDataScoreGenerate,
    HealthDataScore,
)
from bson import ObjectId
from app.utils.ai.prompts import get_prompts
from app.services.backend_services.nudge_service import get_nudge_service
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from app.services.backend_services.encryption_service import get_encryption_service
import logging
from app.exceptions import (
    UserNotFoundError,
    TokenNotFoundError,
    NotificationDisabledError,
    NotificationError,
)

logger = logging.getLogger(__name__)


class HealthAlertService:
    def __init__(self):
        self.db = get_db()
        self.health_data_collection = self.db["health_data"]
        self.health_alert_collection = self.db["health_alerts"]
        self.nudge_service = get_nudge_service()
        self.prompts = get_prompts()
        self.encryption_service = get_encryption_service()
        # TODO: Create a central instance to use ChatOpenAI service
        MAX_RETRIES = 5
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_retries=MAX_RETRIES,
        )

    def _calculate_aggregated_summary(
        self, hourly_data: List[HealthMetricHourlyData]
    ) -> AggregatedHealthSummary:
        total_steps = sum(item.data.steps.value for item in hourly_data)
        heart_rates = [item.data.heartRate.value for item in hourly_data]
        active_energy = sum(item.data.activeEnergy.value for item in hourly_data)
        total_sleep = sum(item.data.sleep.value for item in hourly_data)
        weights = [item.data.weight.value for item in hourly_data]
        body_fats = [item.data.bodyFat.value for item in hourly_data]
        blood_glucose_values = [item.data.bloodGlucose.value for item in hourly_data]
        oxygen_saturation_values = [
            item.data.oxygenSaturation.value for item in hourly_data
        ]

        step_summary = StepSummary(total=total_steps)
        heart_rate_summary = HeartRateSummary(
            average=sum(heart_rates) / len(heart_rates) if heart_rates else 0,
            max=max(heart_rates) if heart_rates else 0,
            min=min(heart_rates) if heart_rates else 0,
        )
        active_energy_summary = ActiveEnergySummary(total=active_energy)
        sleep_summary = SleepSummary(totalHours=total_sleep)
        weight_summary = WeightSummary(
            value=sum(weights) / len(weights) if weights else 0,
            unit=hourly_data[0].data.weight.unit if hourly_data else "kg",
        )
        body_fat_summary = BodyFatSummary(
            value=sum(body_fats) / len(body_fats) if body_fats else 0,
            unit=hourly_data[0].data.bodyFat.unit if hourly_data else "%",
        )
        blood_glucose_summary = BloodGlucoseSummary(
            average=(
                sum(blood_glucose_values) / len(blood_glucose_values)
                if blood_glucose_values
                else 0
            ),
            max=max(blood_glucose_values) if blood_glucose_values else 0,
            min=min(blood_glucose_values) if blood_glucose_values else 0,
        )
        oxygen_saturation_summary = OxygenSaturationSummary(
            average=(
                sum(oxygen_saturation_values) / len(oxygen_saturation_values)
                if oxygen_saturation_values
                else 0
            ),
            max=max(oxygen_saturation_values) if oxygen_saturation_values else 0,
            min=min(oxygen_saturation_values) if oxygen_saturation_values else 0,
        )

        aggregated_summary_dict = {
            "date": datetime.now(timezone.utc),
            "step": step_summary,
            "heartRate": heart_rate_summary,
            "activeEnergy": active_energy_summary,
            "sleep": sleep_summary,
            "weight": weight_summary,
            "bodyFat": body_fat_summary,
            "bloodGlucose": blood_glucose_summary,
            "oxygenSaturation": oxygen_saturation_summary,
        }

        return AggregatedHealthSummary(**aggregated_summary_dict)

    def _generate_aggregated_summary_with_single_health_metric_data(
        self, data: HealthMetricData
    ) -> AggregatedHealthSummary:
        aggregated_summary_dict = {
            "date": datetime.now(timezone.utc),
            "step": {"total": data.steps.value},
            "heartRate": {
                "average": data.heartRate.value,
                "max": data.heartRate.value,
                "min": data.heartRate.value,
            },
            "activeEnergy": {"total": data.activeEnergy.value},
            "sleep": {"totalHours": data.sleep.value},
            "weight": {"value": data.weight.value, "unit": data.weight.unit},
            "bodyFat": {"value": data.bodyFat.value, "unit": data.bodyFat.unit},
            "bloodGlucose": {
                "average": data.bloodGlucose.value,
                "max": data.bloodGlucose.value,
                "min": data.bloodGlucose.value,
            },
            "oxygenSaturation": {
                "average": data.oxygenSaturation.value,
                "max": data.oxygenSaturation.value,
                "min": data.oxygenSaturation.value,
            },
        }
        return AggregatedHealthSummary(**aggregated_summary_dict)

    async def get_latest_health_data_by_user_email(self, user_email: str) -> Optional[HealthData]:
        now_utc = datetime.now(timezone.utc)
        start_of_day = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
        health_data_dict = await self.health_data_collection.find_one(
            {
                "user_email": user_email,
                "created_at": {"$gte": start_of_day, "$lte": end_of_day},
            }
        )
        if not health_data_dict:
            return None
        health_data = HealthData(
            id=str(health_data_dict["_id"]),
            created_at=health_data_dict["created_at"],
            user_email=health_data_dict["user_email"],
            hourly_data=[
                HealthMetricHourlyData(**item)
                for item in health_data_dict.get("hourly_data", [])
            ],
            aggregated_summary=AggregatedHealthSummary(
                **health_data_dict.get("aggregated_summary", {})
            ),
        )
        return health_data

    # TODO: Use get_latest_health_data_by_user_email instead of manually querying the DB
    # TODO: Update updated_at field in the health data document
    async def store_hourly_health_data(
        self, user_email: str, data: HealthMetricData
    ) -> HealthMetricHourlyData:
        now_utc = datetime.now(timezone.utc)
        start_of_day = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
        health_data_dict = await self.health_data_collection.find_one(
            {
                "user_email": user_email,
                "created_at": {"$gte": start_of_day, "$lte": end_of_day},
            }
        )
        if not health_data_dict:
            aggregated_summary = (
                self._generate_aggregated_summary_with_single_health_metric_data(data)
            )
            health_metric_hourly_data = HealthMetricHourlyData(
                data=data, created_at=datetime.now(timezone.utc)
            )
            health_data_dict = {
                "user_email": user_email,
                "created_at": datetime.now(timezone.utc),
                "hourly_data": [health_metric_hourly_data.model_dump()],
                "aggregated_summary": aggregated_summary.model_dump(),
            }
            health_data_create = HealthDataCreate(**health_data_dict)
            insertion = await self.health_data_collection.insert_one(
                health_data_create.model_dump()
            )
            health_data = HealthData(
                id=str(insertion.inserted_id),
                created_at=health_data_create.created_at,
                user_email=health_data_create.user_email,
                hourly_data=health_data_create.hourly_data,
                aggregated_summary=health_data_create.aggregated_summary,
            )
        else:
            health_data_dict["id"] = str(health_data_dict["_id"])
            del health_data_dict["_id"]

            # CRITICAL FIX: Decrypt hourly_data from database before using it
            # The hourly_data in the database has encrypted values that need to be decrypted
            prev_health_metric_hourly_data_list = []
            for health_metric_hourly_data_dict in health_data_dict["hourly_data"]:
                # Create HealthMetricHourlyData object from dict (values may still be encrypted)
                hourly_data_obj = HealthMetricHourlyData(**health_metric_hourly_data_dict)
                # Decrypt the hourly_data to get actual values
                decrypted_hourly_data = self.encryption_service.decrypt_document(
                    hourly_data_obj, HealthMetricHourlyData
                )
                prev_health_metric_hourly_data_list.append(decrypted_hourly_data)

            health_metric_hourly_data = HealthMetricHourlyData(
                data=data, created_at=datetime.now(timezone.utc)
            )

            health_metric_hourly_data_list = prev_health_metric_hourly_data_list + [
                health_metric_hourly_data
            ]

            aggregated_summary = self._calculate_aggregated_summary(
                health_metric_hourly_data_list
            )

            health_data = HealthData(
                id=health_data_dict["id"],
                created_at=health_data_dict["created_at"],
                user_email=health_data_dict["user_email"],
                hourly_data=health_metric_hourly_data_list,
                aggregated_summary=aggregated_summary,
            )

            await self.health_data_collection.update_one(
                {"user_email": user_email},
                {
                    "$push": {"hourly_data": health_metric_hourly_data.model_dump()},
                    "$set": {"aggregated_summary": aggregated_summary.model_dump()},
                },
            )

        # TEMPORARILY DISABLED: Critical alerts generation and notifications
        # health_alerts = await self._generate_health_alerts(health_data)
        # for alert_generate in health_alerts:
        #     alert = HealthAlertCreate(
        #         **alert_generate.model_dump(),
        #         user_email=user_email,
        #         health_data_id=health_data.id,
        #         status=AlertStatus.ACTIVE,
        #         created_at=datetime.now(timezone.utc),
        #     )
        #     alert = self.encryption_service.encrypt_document(alert, HealthAlertCreate)
        #     insertion_id = await self.health_alert_collection.insert_one(alert.model_dump())
        #     alert = HealthAlert(
        #         id=str(insertion_id.inserted_id), **alert.model_dump()
        #     )
        #     alert = self.encryption_service.decrypt_document(alert, HealthAlert)
        #     
        #     try:
        #         await self.nudge_service.send_fcm_notification(
        #             email=user_email,
        #             title=alert.title,
        #             body=alert.message,
        #         )
        #         await self.nudge_service._update_notification_cooldown(user_email, "health_alert")
        #         logger.info(f"FCM notification sent successfully for health alert to {user_email}")
        #     except UserNotFoundError as e:
        #         logger.warning(f"Cannot send FCM notification: User not found: {user_email}")
        #     except TokenNotFoundError as e:
        #         logger.warning(f"Cannot send FCM notification: No FCM token registered for {user_email}. User needs to register device token.")
        #     except NotificationDisabledError as e:
        #         logger.info(f"Cannot send FCM notification: Notifications disabled for {user_email}")
        #     except NotificationError as e:
        #         error_type = e.details.get("error_type", "unknown") if hasattr(e, "details") else "unknown"
        #         if error_type == "invalid_token":
        #             # Invalid token - user needs to re-register, but don't crash the alert creation
        #             logger.warning(
        #                 f"Health alert created but FCM notification failed due to invalid token for {user_email}. "
        #                 "Alert is still created and stored. User needs to re-register device token."
        #             )
        #         else:
        #             # Other Firebase errors - log with full details
        #             logger.error(
        #                 f"Firebase notification error for {user_email}: {str(e)}",
        #                 exc_info=True
        #             )
        #     except Exception as e:
        #         logger.error(
        #             f"Unexpected error sending FCM notification for health alert to {user_email}: {type(e).__name__}: {str(e)}",
        #             exc_info=True
        #         )
        logger.info(f"[HEALTH_ALERT] Critical alerts generation temporarily disabled for {user_email}")

        return health_metric_hourly_data

    async def _get_previous_health_alerts(
        self, health_data_id: str
    ) -> List[HealthAlert]:
        previous_alerts = await self.health_alert_collection.find(
            {"health_data_id": health_data_id, "status": AlertStatus.ACTIVE.value}
        ).to_list(length=None)
        for alert in previous_alerts:
            alert["id"] = str(alert["_id"])
            del alert["_id"]
        previous_alerts = [HealthAlert(**alert) for alert in previous_alerts]
        previous_alerts = [
            self.encryption_service.decrypt_document(alert, HealthAlert)
            for alert in previous_alerts
        ]
        return previous_alerts

    def _check_threshold_violations(
        self, hourly_data: List[HealthMetricHourlyData], aggregated_summary: Optional[AggregatedHealthSummary] = None
    ) -> List[HealthAlertGenerate]:
        """
        Check hourly data against predefined thresholds and generate alerts for violations.
        Returns a list of HealthAlertGenerate objects for threshold violations.
        Deduplicates alerts by metric to prevent multiple alerts for the same metric.
        
        For sleep, uses aggregated total sleep hours instead of individual hourly values,
        since sleep is cumulative and should be evaluated as a daily total.
        """
        threshold_alerts = []
        alerted_metrics = set()  # Track metrics that have already triggered an alert
        
        # Define thresholds
        THRESHOLDS = {
            "steps": {"min": 100, "max": 30000},
            "heartRate": {"min": 50, "max": 120},
            "activeEnergy": {"min": 30, "max": 600},
            "sleep": {"min": 4, "max": 11},
            "weight": {"min": 17, "max": 30},
            "bodyFat": {"min": 5, "max": 40},
            "bloodGlucose": {"min": 60, "max": 180},
            "oxygenSaturation": {"min": 90, "max": 100},
        }
        
        # Metric to unit mapping
        METRIC_UNITS = {
            "steps": "steps",
            "heartRate": "bpm",
            "activeEnergy": "kcal/hour",
            "sleep": "hours",
            "weight": "kg",
            "bodyFat": "%",
            "bloodGlucose": "mg/dL",
            "oxygenSaturation": "%",
        }
        
        # Check sleep using aggregated total (if available) - sleep is cumulative and should be evaluated as daily total
        # Only check sleep alerts if we have sufficient data (>= 3 hours) or it's after 10 AM with at least 1 hour of data
        # This prevents false alerts when hourly data is submitted early in the day with minimal sleep data
        now_utc = datetime.now(timezone.utc)
        should_check_sleep = False
        if aggregated_summary and aggregated_summary.sleep and aggregated_summary.sleep.totalHours is not None:
            total_sleep = aggregated_summary.sleep.totalHours
            # Check if we have enough sleep data (>= 3 hours indicates a reasonable sleep period)
            # OR it's late enough in the day (after 10 AM) with at least 1 hour of data
            # This prevents false alerts from early hourly submissions while still catching real issues
            has_sufficient_data = total_sleep >= 3.0
            is_late_enough_with_data = now_utc.hour >= 10 and total_sleep >= 1.0
            should_check_sleep = has_sufficient_data or is_late_enough_with_data
        
        if should_check_sleep and "sleep" not in alerted_metrics:
            total_sleep = aggregated_summary.sleep.totalHours
            if total_sleep < THRESHOLDS["sleep"]["min"]:
                threshold_alerts.append(HealthAlertGenerate(
                    metric="sleep",
                    title="Insufficient Sleep Alert",
                    key_point=f"Sleep ({total_sleep} hours) is below minimum threshold ({THRESHOLDS['sleep']['min']} hours)",
                    message=f"Your sleep duration of {total_sleep} hours is below the recommended minimum of {THRESHOLDS['sleep']['min']} hours. Adequate sleep is crucial for health.",
                    severity=HealthAlertSeverity.MEDIUM,
                    current_value=total_sleep,
                    threshold_value=THRESHOLDS["sleep"]["min"],
                    unit=METRIC_UNITS["sleep"]
                ))
                alerted_metrics.add("sleep")
            elif total_sleep > THRESHOLDS["sleep"]["max"]:
                threshold_alerts.append(HealthAlertGenerate(
                    metric="sleep",
                    title="Excessive Sleep Alert",
                    key_point=f"Sleep ({total_sleep} hours) exceeds maximum threshold ({THRESHOLDS['sleep']['max']} hours)",
                    message=f"Your sleep duration of {total_sleep} hours exceeds the typical maximum of {THRESHOLDS['sleep']['max']} hours. Excessive sleep may indicate underlying health issues.",
                    severity=HealthAlertSeverity.MEDIUM,
                    current_value=total_sleep,
                    threshold_value=THRESHOLDS["sleep"]["max"],
                    unit=METRIC_UNITS["sleep"]
                ))
                alerted_metrics.add("sleep")
        
        # Check each hourly data entry (skip sleep as it's handled above with aggregated data)
        for hourly_entry in hourly_data:
            if not hourly_entry.data:
                continue
                
            data = hourly_entry.data
            
            # Check steps
            if data.steps.isAvailable and data.steps.value is not None and "steps" not in alerted_metrics:
                if data.steps.value < THRESHOLDS["steps"]["min"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="steps",
                        title="Low Step Count Alert",
                        key_point=f"Steps ({data.steps.value}) are below minimum threshold ({THRESHOLDS['steps']['min']})",
                        message=f"Your step count of {data.steps.value} steps is below the healthy minimum of {THRESHOLDS['steps']['min']} steps. Consider increasing your daily activity.",
                        severity=HealthAlertSeverity.MEDIUM,
                        current_value=data.steps.value,
                        threshold_value=THRESHOLDS["steps"]["min"],
                        unit=METRIC_UNITS["steps"]
                    ))
                    alerted_metrics.add("steps")
                elif data.steps.value > THRESHOLDS["steps"]["max"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="steps",
                        title="Excessive Step Count Alert",
                        key_point=f"Steps ({data.steps.value}) exceed maximum threshold ({THRESHOLDS['steps']['max']})",
                        message=f"Your step count of {data.steps.value} steps exceeds the recommended maximum of {THRESHOLDS['steps']['max']} steps. This may indicate data error or extreme activity.",
                        severity=HealthAlertSeverity.LOW,
                        current_value=data.steps.value,
                        threshold_value=THRESHOLDS["steps"]["max"],
                        unit=METRIC_UNITS["steps"]
                    ))
                    alerted_metrics.add("steps")
            
            # Check heartRate
            if data.heartRate.isAvailable and data.heartRate.value is not None and "heartRate" not in alerted_metrics:
                if data.heartRate.value < THRESHOLDS["heartRate"]["min"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="heartRate",
                        title="Low Heart Rate Alert",
                        key_point=f"Heart rate ({data.heartRate.value} bpm) is below minimum threshold ({THRESHOLDS['heartRate']['min']} bpm)",
                        message=f"Your heart rate of {data.heartRate.value} bpm is below the normal minimum of {THRESHOLDS['heartRate']['min']} bpm. Please consult with a healthcare provider.",
                        severity=HealthAlertSeverity.HIGH,
                        current_value=data.heartRate.value,
                        threshold_value=THRESHOLDS["heartRate"]["min"],
                        unit=METRIC_UNITS["heartRate"]
                    ))
                    alerted_metrics.add("heartRate")
                elif data.heartRate.value > THRESHOLDS["heartRate"]["max"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="heartRate",
                        title="High Heart Rate Alert",
                        key_point=f"Heart rate ({data.heartRate.value} bpm) exceeds maximum threshold ({THRESHOLDS['heartRate']['max']} bpm)",
                        message=f"Your heart rate of {data.heartRate.value} bpm exceeds the normal maximum of {THRESHOLDS['heartRate']['max']} bpm. Please monitor and consider consulting a healthcare provider if this persists.",
                        severity=HealthAlertSeverity.HIGH,
                        current_value=data.heartRate.value,
                        threshold_value=THRESHOLDS["heartRate"]["max"],
                        unit=METRIC_UNITS["heartRate"]
                    ))
                    alerted_metrics.add("heartRate")
            
            # Check activeEnergy
            if data.activeEnergy.isAvailable and data.activeEnergy.value is not None and "activeEnergy" not in alerted_metrics:
                if data.activeEnergy.value < THRESHOLDS["activeEnergy"]["min"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="activeEnergy",
                        title="Low Active Energy Alert",
                        key_point=f"Active energy ({data.activeEnergy.value} kcal/hour) is below minimum threshold ({THRESHOLDS['activeEnergy']['min']} kcal/hour)",
                        message=f"Your active energy of {data.activeEnergy.value} kcal/hour is below the healthy minimum of {THRESHOLDS['activeEnergy']['min']} kcal/hour. Consider increasing your activity level.",
                        severity=HealthAlertSeverity.LOW,
                        current_value=data.activeEnergy.value,
                        threshold_value=THRESHOLDS["activeEnergy"]["min"],
                        unit=METRIC_UNITS["activeEnergy"]
                    ))
                    alerted_metrics.add("activeEnergy")
                elif data.activeEnergy.value > THRESHOLDS["activeEnergy"]["max"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="activeEnergy",
                        title="Excessive Active Energy Alert",
                        key_point=f"Active energy ({data.activeEnergy.value} kcal/hour) exceeds maximum threshold ({THRESHOLDS['activeEnergy']['max']} kcal/hour)",
                        message=f"Your active energy of {data.activeEnergy.value} kcal/hour exceeds the typical maximum of {THRESHOLDS['activeEnergy']['max']} kcal/hour. This may indicate intense activity or data error.",
                        severity=HealthAlertSeverity.LOW,
                        current_value=data.activeEnergy.value,
                        threshold_value=THRESHOLDS["activeEnergy"]["max"],
                        unit=METRIC_UNITS["activeEnergy"]
                    ))
                    alerted_metrics.add("activeEnergy")
            
            # Sleep is checked above using aggregated total, skip individual hourly sleep checks
            
            if data.weight.isAvailable and data.weight.value is not None:
                # Skipping weight threshold check - BMI calculation requires height which is not available
                pass
            
            # Check bodyFat
            if data.bodyFat.isAvailable and data.bodyFat.value is not None and "bodyFat" not in alerted_metrics:
                if data.bodyFat.value < THRESHOLDS["bodyFat"]["min"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="bodyFat",
                        title="Low Body Fat Alert",
                        key_point=f"Body fat ({data.bodyFat.value}%) is below minimum threshold ({THRESHOLDS['bodyFat']['min']}%)",
                        message=f"Your body fat percentage of {data.bodyFat.value}% is below the healthy minimum of {THRESHOLDS['bodyFat']['min']}%. Very low body fat can impact health.",
                        severity=HealthAlertSeverity.MEDIUM,
                        current_value=data.bodyFat.value,
                        threshold_value=THRESHOLDS["bodyFat"]["min"],
                        unit=METRIC_UNITS["bodyFat"]
                    ))
                    alerted_metrics.add("bodyFat")
                elif data.bodyFat.value > THRESHOLDS["bodyFat"]["max"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="bodyFat",
                        title="High Body Fat Alert",
                        key_point=f"Body fat ({data.bodyFat.value}%) exceeds maximum threshold ({THRESHOLDS['bodyFat']['max']}%)",
                        message=f"Your body fat percentage of {data.bodyFat.value}% exceeds the recommended maximum of {THRESHOLDS['bodyFat']['max']}%. Consider consulting with a healthcare provider about healthy weight management.",
                        severity=HealthAlertSeverity.MEDIUM,
                        current_value=data.bodyFat.value,
                        threshold_value=THRESHOLDS["bodyFat"]["max"],
                        unit=METRIC_UNITS["bodyFat"]
                    ))
                    alerted_metrics.add("bodyFat")
            
            # Check bloodGlucose
            if data.bloodGlucose.isAvailable and data.bloodGlucose.value is not None and "bloodGlucose" not in alerted_metrics:
                if data.bloodGlucose.value < THRESHOLDS["bloodGlucose"]["min"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="bloodGlucose",
                        title="Low Blood Glucose Alert",
                        key_point=f"Blood glucose ({data.bloodGlucose.value} mg/dL) is below minimum threshold ({THRESHOLDS['bloodGlucose']['min']} mg/dL)",
                        message=f"Your blood glucose level of {data.bloodGlucose.value} mg/dL is below the normal minimum of {THRESHOLDS['bloodGlucose']['min']} mg/dL. This may indicate hypoglycemia. Please consult with a healthcare provider.",
                        severity=HealthAlertSeverity.HIGH,
                        current_value=data.bloodGlucose.value,
                        threshold_value=THRESHOLDS["bloodGlucose"]["min"],
                        unit=METRIC_UNITS["bloodGlucose"]
                    ))
                    alerted_metrics.add("bloodGlucose")
                elif data.bloodGlucose.value > THRESHOLDS["bloodGlucose"]["max"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="bloodGlucose",
                        title="High Blood Glucose Alert",
                        key_point=f"Blood glucose ({data.bloodGlucose.value} mg/dL) exceeds maximum threshold ({THRESHOLDS['bloodGlucose']['max']} mg/dL)",
                        message=f"Your blood glucose level of {data.bloodGlucose.value} mg/dL exceeds the normal maximum of {THRESHOLDS['bloodGlucose']['max']} mg/dL. Please monitor and consult with a healthcare provider.",
                        severity=HealthAlertSeverity.HIGH,
                        current_value=data.bloodGlucose.value,
                        threshold_value=THRESHOLDS["bloodGlucose"]["max"],
                        unit=METRIC_UNITS["bloodGlucose"]
                    ))
                    alerted_metrics.add("bloodGlucose")
            
            # Check oxygenSaturation
            if data.oxygenSaturation.isAvailable and data.oxygenSaturation.value is not None and "oxygenSaturation" not in alerted_metrics:
                if data.oxygenSaturation.value < THRESHOLDS["oxygenSaturation"]["min"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="oxygenSaturation",
                        title="Low Oxygen Saturation Alert",
                        key_point=f"Oxygen saturation ({data.oxygenSaturation.value}%) is below minimum threshold ({THRESHOLDS['oxygenSaturation']['min']}%)",
                        message=f"Your oxygen saturation of {data.oxygenSaturation.value}% is below the normal minimum of {THRESHOLDS['oxygenSaturation']['min']}%. Please seek immediate medical attention if symptoms persist.",
                        severity=HealthAlertSeverity.HIGH,
                        current_value=data.oxygenSaturation.value,
                        threshold_value=THRESHOLDS["oxygenSaturation"]["min"],
                        unit=METRIC_UNITS["oxygenSaturation"]
                    ))
                    alerted_metrics.add("oxygenSaturation")
                elif data.oxygenSaturation.value > THRESHOLDS["oxygenSaturation"]["max"]:
                    threshold_alerts.append(HealthAlertGenerate(
                        metric="oxygenSaturation",
                        title="High Oxygen Saturation Alert",
                        key_point=f"Oxygen saturation ({data.oxygenSaturation.value}%) exceeds maximum threshold ({THRESHOLDS['oxygenSaturation']['max']}%)",
                        message=f"Your oxygen saturation of {data.oxygenSaturation.value}% exceeds the normal maximum of {THRESHOLDS['oxygenSaturation']['max']}%. This may indicate a measurement error.",
                        severity=HealthAlertSeverity.LOW,
                        current_value=data.oxygenSaturation.value,
                        threshold_value=THRESHOLDS["oxygenSaturation"]["max"],
                        unit=METRIC_UNITS["oxygenSaturation"]
                    ))
                    alerted_metrics.add("oxygenSaturation")
        
        return threshold_alerts

    async def _generate_health_alerts(
        self, health_data: HealthData
    ) -> List[HealthAlertGenerate]:
        # TODO: Handle duplicate alerts based on previous active health alerts
        health_data_id = health_data.id

        threshold_alerts = self._check_threshold_violations(
            health_data.hourly_data, 
            aggregated_summary=health_data.aggregated_summary
        )
        previous_health_alerts = await self._get_previous_health_alerts(health_data_id)
        
        # Create a map of metric -> threshold_alert for value enrichment after LLM filtering
        threshold_alerts_map = {
            (alert.metric.value if hasattr(alert.metric, "value") else str(alert.metric)): alert
            for alert in threshold_alerts
        }
        
        prompt = self.prompts.get_health_alerts_prompt(
            threshold_alerts=[alert.model_dump() for alert in threshold_alerts],
            previous_health_alerts=[
                alert.model_dump() for alert in previous_health_alerts
            ],
        )
        # TODO: You can shift this to prompt.py file instead
        system_prompt = "You are a health monitoring AI that generates structured health alerts based on health metrics."
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        structured_llm = self.llm.with_structured_output(HealthAlertGenerationResponse)
        response = await structured_llm.ainvoke(messages)
        
        if not response.should_generate_alert:
            return []
        
        # CRITICAL FIX: Enrich LLM-filtered alerts with actual metric values from threshold_alerts
        # The LLM filters alerts but doesn't preserve the actual values, so we need to match them back
        enriched_alerts = []
        for llm_alert in response.alerts:
            # Find the original threshold alert with the same metric to get actual values
            metric_key = llm_alert.metric.value if hasattr(llm_alert.metric, "value") else str(llm_alert.metric)
            original_alert = threshold_alerts_map.get(metric_key)
            if original_alert:
                # Create enriched alert with actual values from uploaded health data
                enriched_alert = HealthAlertGenerate(
                    metric=llm_alert.metric,
                    title=llm_alert.title,
                    key_point=llm_alert.key_point,
                    message=llm_alert.message,
                    severity=llm_alert.severity,
                    current_value=original_alert.current_value,
                    threshold_value=original_alert.threshold_value,
                    unit=original_alert.unit
                )
                enriched_alerts.append(enriched_alert)
            else:
                # If no match found, use LLM alert as-is (shouldn't happen, but safe fallback)
                logger.warning(f"Could not find original threshold alert for metric: {llm_alert.metric}")
                enriched_alerts.append(llm_alert)
        
        return enriched_alerts

    async def get_active_health_alerts(self, user_email: str) -> List[HealthAlert]:
        # Auto-resolve alerts older than 2 days
        await self._auto_resolve_old_alerts(user_email, days=2)
        
        # Filter to only alerts from the last 2 hours
        now_utc = datetime.now(timezone.utc)
        two_hours_ago = now_utc - timedelta(hours=2)
        
        active_alerts = await self.health_alert_collection.find(
            {
                "user_email": user_email, 
                "status": AlertStatus.ACTIVE.value,
                "created_at": {"$gte": two_hours_ago, "$lte": now_utc}  # Only alerts from the last 2 hours
            }
        ).sort("created_at", -1).to_list(length=None)  # Sort by created_at descending (newest first)
        
        for alert in active_alerts:
            alert["id"] = str(alert["_id"])
            del alert["_id"]
        model_alerts = [HealthAlert(**alert) for alert in active_alerts]
        alerts = [
            self.encryption_service.decrypt_document(alert, HealthAlert)
            for alert in model_alerts
        ]
        
        # Deduplicate: Keep only the latest alert for each metric type
        # Since alerts are sorted by created_at descending, first occurrence is the latest
        metric_to_latest_alert = {}
        for alert in alerts:
            metric = alert.metric.value if hasattr(alert.metric, 'value') else str(alert.metric)
            # Only add if we haven't seen this metric yet (first = latest due to sort)
            if metric not in metric_to_latest_alert:
                metric_to_latest_alert[metric] = alert
        
        # Return only the latest alert for each metric, sorted by created_at descending
        deduplicated_alerts = list(metric_to_latest_alert.values())
        deduplicated_alerts.sort(key=lambda x: x.created_at, reverse=True)
        
        return deduplicated_alerts
    
    async def _auto_resolve_old_alerts(self, user_email: str, days: int = 2) -> int:
        """Automatically resolve alerts older than specified days for a user"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        result = await self.health_alert_collection.update_many(
            {
                "user_email": user_email,
                "status": AlertStatus.ACTIVE.value,
                "created_at": {"$lt": cutoff_date}
            },
            {"$set": {"status": AlertStatus.RESOLVED.value}}
        )
        if result.modified_count > 0:
            logger.info(f"Auto-resolved {result.modified_count} old alerts for user {user_email}")
        return result.modified_count
    
    async def score_health_data(self, health_data: HealthData) -> HealthDataScore:
        prompt = self.prompts.get_health_data_score_prompt(health_data)
        structured_llm = self.llm.with_structured_output(HealthDataScoreGenerate)
        # Use proper LangChain message format for consistency
        messages = [HumanMessage(content=prompt)]
        response = await structured_llm.ainvoke(messages)
        health_data_score = HealthDataScore(
            health_data_id=health_data.id,
            score=response.score,
            reasons=response.reasons
        )
        return health_data_score

    async def mark_health_alert_resolve(self, health_alert_id: str) -> bool:
        result = await self.health_alert_collection.update_one(
            {"_id": ObjectId(health_alert_id)},
            {"$set": {"status": AlertStatus.RESOLVED.value}},
        )
        return result.modified_count > 0


def get_health_alert_service():
    return HealthAlertService()
