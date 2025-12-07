"""
Personalization Profile Service

This service generates personalized user profiles using a fine-tuned LLM.
The profile includes 7 clusters:
1. User Goals & Therapeutic Focus
2. Lifestyle Pillars
3. Nutrition Plan Tiering
4. Personalized Action Mapping
5. Supplement Plan
6. Tiered Insight Delivery
7. Cross-Validated Logic

The profile is generated once and cached in the user document.
It's regenerated when:
- User data changes significantly
- Manual regeneration is requested
"""

import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
import os

from app.config import OPENAI_API_KEY, PERSONALIZATION_MODEL
from app.services.backend_services.db import get_db
from app.services.ai_services.lab_report_service import get_lab_report_service
from app.services.backend_services.health_alert_service import get_health_alert_service

logger = logging.getLogger(__name__)


class PersonalizationProfileService:
    """
    Service to generate and manage personalized user profiles using fine-tuned LLM.
    """
    
    def __init__(self):
        """Initialize the personalization profile service."""
        self.db = get_db()
        self.user_collection = self.db["users"]
        self.preferences_collection = self.db["preferences"]
        self.lab_report_service = get_lab_report_service()
        self.health_alert_service = get_health_alert_service()
        
        self.fine_tuned_model = PERSONALIZATION_MODEL
        
        # Use OPENAI_API_KEY for fine-tuned model
        self.llm = ChatOpenAI(
            model=self.fine_tuned_model,
            openai_api_key=OPENAI_API_KEY,
            temperature=1,  # Fine-tuned models only support default temperature
            max_retries=3,
        )
        
        logger.info(f"PersonalizationProfileService initialized with model: {self.fine_tuned_model}")
    
    async def should_generate_profile(self, user_email: str) -> tuple[bool, Optional[str]]:
        """
        Check if we should generate a personalization profile for the user.        
        """
        has_health_data = await self._user_has_health_data(user_email)
        if not has_health_data:
            return False, "User does not have health data (Health Connect not enabled or no data)"
        
        has_lab_reports = await self._user_has_lab_reports(user_email)
        if not has_lab_reports:
            return False, "User has not uploaded any lab reports"
        
        user = await self.user_collection.find_one({"email": user_email})
        if not user:
            return False, "User not found"
        
        if "personalization_profile" not in user or not user["personalization_profile"]:
            return True, "No personalization profile exists"
        
        return False, "Valid profile already exists"
    
    async def _user_has_health_data(self, user_email: str) -> bool:
        """Check if user has any health data in the system (not just today's)."""
        try:
            health_data_collection = self.db["health_data"]
            health_data_dict = await health_data_collection.find_one(
                {"user_email": user_email},
                sort=[("created_at", -1)] 
            )
            
            if not health_data_dict:
                return False
            
            has_hourly = health_data_dict.get("hourly_data") and len(health_data_dict.get("hourly_data", [])) > 0
            has_summary = health_data_dict.get("aggregated_summary") is not None
            
            return has_hourly or has_summary
            
        except Exception as e:
            logger.error(f"Error checking health data for {user_email}: {e}")
            return False
    
    async def _user_has_lab_reports(self, user_email: str) -> bool:
        """Check if user has uploaded at least one lab report."""
        try:
            lab_reports = await self.lab_report_service.get_lab_reports_by_user(user_email)
            return lab_reports and len(lab_reports) > 0
        except Exception as e:
            logger.error(f"Error checking lab reports for {user_email}: {e}", exc_info=True)
            raise
    
    async def get_cached_profile(self, user_email: str) -> Optional[Dict[str, Any]]:
        """
        Get cached personalization profile from user document.
        """
        user = await self.user_collection.find_one({"email": user_email})
        if user and "personalization_profile" in user:
            return user["personalization_profile"]
        return None
    
    async def generate_profile(self, user_email: str, force: bool = False) -> Dict[str, Any]:
        """
        Generate personalization profile for a user using the fine-tuned LLM.
        """
        if not force:
            should_generate, reason = await self.should_generate_profile(user_email)
            if not should_generate:
                cached = await self.get_cached_profile(user_email)
                if cached:
                    return cached
                raise ValueError(f"Cannot generate profile: {reason}")
        
        user_data = await self._gather_user_data(user_email)
        
        try:
            profile_data = await self._generate_with_llm(user_data)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise
        
        profile = {
            "generated_at": datetime.now(timezone.utc),
            "data": profile_data
        }
        
        await self._save_profile(user_email, profile)
        
        logger.info(f"Profile generated successfully for {user_email}")
        return profile
    
    async def _gather_user_data(self, user_email: str) -> Dict[str, Any]:
        """
        Gather all relevant user data for profile generation.
        
        """
        user = await self.user_collection.find_one({"email": user_email})
        preferences = await self.preferences_collection.find_one({"email": user_email})
        
        health_data = await self.health_alert_service.get_latest_health_data_by_user_email(user_email)
        
        lab_reports = await self.lab_report_service.get_lab_reports_by_user(user_email)
        detailed_reports = []
        for report_summary in (lab_reports if lab_reports else []):
            try:
                detailed = await self.lab_report_service.get_lab_report_by_id(
                    report_summary.id, user_email
                )
                if detailed:
                    detailed_reports.append(detailed)
            except Exception as e:
                logger.warning(f"Error fetching detailed lab report {report_summary.id}: {e}")
                continue
        
        user_data = {
            "user_info": {
                "name": user.get("name", ""),
                "email": user_email,
                "age": preferences.get("age") if preferences else None,
                "gender": preferences.get("gender") if preferences else None,
                "sex": preferences.get("gender") if preferences else None,  # Use gender as sex for compatibility
                "timezone": user.get("timezone", "UTC"),
            },
            "preferences": {
                "health_goals": preferences.get("healthGoals", []) if preferences else [],
                "goals": preferences.get("healthGoals", []) if preferences else [],  # Alias for compatibility
                "conditions": preferences.get("conditions", []) if preferences else [],
                "diagnoses": preferences.get("conditions", []) if preferences else [],  # Alias for compatibility
                "at_risk_conditions": preferences.get("atRiskConditions", []) if preferences else [],
                "communication_style": preferences.get("communicationStyle", "balanced") if preferences else "balanced",
            },
            "health_data": self._format_health_data(health_data) if health_data else None,
            "lab_reports": self._format_lab_reports(detailed_reports),
        }
        
        # Add weight from health_data if available (height not in health_data schema)
        if health_data:
            # Try aggregated_summary first
            if health_data.aggregated_summary and health_data.aggregated_summary.weight:
                weight_value = health_data.aggregated_summary.weight.value
                if weight_value and weight_value > 0:
                    user_data["user_info"]["weight"] = weight_value
            # Fallback to latest hourly data
            elif health_data.hourly_data and len(health_data.hourly_data) > 0:
                latest = health_data.hourly_data[-1]
                if latest.data and latest.data.weight and latest.data.weight.value:
                    weight_value = latest.data.weight.value
                    if weight_value and weight_value > 0:
                        user_data["user_info"]["weight"] = weight_value
        
        return user_data
    
    def _format_health_data(self, health_data) -> Dict[str, Any]:
        """Format health data for LLM consumption."""
        if not health_data:
            return {}
        
        formatted = {}
        
        # Use aggregated_summary if available (more representative)
        if health_data.aggregated_summary:
            summary = health_data.aggregated_summary
            if summary.step and summary.step.total is not None:
                formatted["steps"] = {"value": summary.step.total, "unit": "steps"}
            if summary.heartRate and summary.heartRate.average is not None:
                formatted["heart_rate"] = {"value": summary.heartRate.average, "unit": "bpm"}
            if summary.activeEnergy and summary.activeEnergy.total is not None:
                formatted["active_energy"] = {"value": summary.activeEnergy.total, "unit": "kcal"}
            if summary.sleep and summary.sleep.totalHours is not None:
                formatted["sleep"] = {"value": summary.sleep.totalHours, "unit": "hours"}
            if summary.weight:
                formatted["weight"] = {"value": summary.weight.value, "unit": summary.weight.unit}
            if summary.bodyFat:
                formatted["body_fat"] = {"value": summary.bodyFat.value, "unit": summary.bodyFat.unit}
            if summary.bloodGlucose and summary.bloodGlucose.average is not None:
                formatted["blood_glucose"] = {"value": summary.bloodGlucose.average, "unit": "mg/dL"}
            if summary.oxygenSaturation and summary.oxygenSaturation.average is not None:
                formatted["oxygen_saturation"] = {"value": summary.oxygenSaturation.average, "unit": "%"}
        
        # Fallback to latest hourly data if aggregated_summary is missing values
        elif health_data.hourly_data and len(health_data.hourly_data) > 0:
            latest = health_data.hourly_data[-1]  # Get most recent hourly data
            if latest.data:
                data = latest.data
                formatted = {
                    "steps": {"value": data.steps.value, "unit": data.steps.unit},
                    "heart_rate": {"value": data.heartRate.value, "unit": data.heartRate.unit},
                    "active_energy": {"value": data.activeEnergy.value, "unit": data.activeEnergy.unit},
                    "sleep": {"value": data.sleep.value, "unit": data.sleep.unit},
                    "weight": {"value": data.weight.value, "unit": data.weight.unit},
                    "body_fat": {"value": data.bodyFat.value, "unit": data.bodyFat.unit},
                    "blood_glucose": {"value": data.bloodGlucose.value, "unit": data.bloodGlucose.unit},
                    "oxygen_saturation": {"value": data.oxygenSaturation.value, "unit": data.oxygenSaturation.unit},
                }
        
        return formatted
    
    def _format_lab_reports(self, lab_reports) -> list[Dict[str, Any]]:
        """Format lab reports for LLM consumption."""
        formatted = []
        for report in lab_reports:
            formatted.append({
                "test_date": str(report.test_date) if hasattr(report, 'test_date') else None,
                "lab_name": report.lab_name if hasattr(report, 'lab_name') else None,
                "properties": [
                    {
                        "property_name": prop.property_name, 
                        "value": prop.value,
                        "unit": prop.unit,
                        "reference_range": prop.reference_range,
                        "status": prop.status,
                    }
                    for prop in (report.properties if hasattr(report, 'properties') else [])
                ]
            })
        return formatted
    
    async def _generate_with_llm(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalization profile using fine-tuned LLM.
        """
        user_info = user_data.get('user_info', {})
        health_data = user_data.get('health_data', {})
        lab_reports = user_data.get('lab_reports', [])
        preferences = user_data.get('preferences', {})
        
        user_profile_parts = []
        
        if user_info:
            user_profile_parts.append("Demographics & Baseline")
            if user_info.get('sex') or user_info.get('gender'):
                user_profile_parts.append(f"Sex: {user_info.get('sex') or user_info.get('gender', '')}")
            if user_info.get('age'):
                user_profile_parts.append(f"Age: {user_info['age']}")
            if user_info.get('weight'):
                user_profile_parts.append(f"Weight: {user_info['weight']} kg")
            if user_info.get('height'):
                user_profile_parts.append(f"Height: {user_info['height']} cm")
            if user_info.get('bmi'):
                user_profile_parts.append(f"BMI: {user_info['bmi']} kg/m2")
            
            diagnoses = preferences.get('diagnoses') or preferences.get('conditions', [])
            if diagnoses:
                diagnoses_str = ', '.join(diagnoses) if isinstance(diagnoses, list) else str(diagnoses)
                user_profile_parts.append(f"Diagnoses: {diagnoses_str}")
            
            goals = preferences.get('goals') or preferences.get('health_goals', [])
            if goals:
                goals_str = ', '.join(goals) if isinstance(goals, list) else str(goals)
                user_profile_parts.append(f"User-Stated Goals: {goals_str}")
        
        if lab_reports:
            user_profile_parts.append("\nLab Results (Sample Data)")
            for report in lab_reports: 
                if report.get('lab_name'):
                    user_profile_parts.append(f"\nLab: {report.get('lab_name')}")
                if report.get('test_date'):
                    user_profile_parts.append(f"Test Date: {report.get('test_date')}")
                if report.get('properties'):
                    for prop in report['properties']:
                        prop_name = prop.get('property_name', '')
                        value = prop.get('value', '')
                        unit = prop.get('unit', '')
                        status = prop.get('status', '')
                        reference_range = prop.get('reference_range', '')
                        if prop_name and value:
                            status_str = f" -> {status}" if status else ""
                            range_str = f" (ref: {reference_range})" if reference_range else ""
                            unit_str = f" {unit}" if unit else ""
                            user_profile_parts.append(f"{prop_name}: {value}{unit_str}{range_str}{status_str}")
        
        if health_data:
            user_profile_parts.append("\nWearable Metrics (Sample Data)")
            if health_data.get('steps') and health_data['steps'].get('value'):
                steps_value = health_data['steps']['value']
                steps_unit = health_data['steps'].get('unit', '')
                user_profile_parts.append(f"Steps per day: ~{steps_value} {steps_unit}".strip())
            if health_data.get('heart_rate') and health_data['heart_rate'].get('value'):
                hr_value = health_data['heart_rate']['value']
                hr_unit = health_data['heart_rate'].get('unit', 'bpm')
                user_profile_parts.append(f"Resting HR: {hr_value} {hr_unit}")
            if health_data.get('sleep') and health_data['sleep'].get('value'):
                sleep_value = health_data['sleep']['value']
                sleep_unit = health_data['sleep'].get('unit', 'hours')
                user_profile_parts.append(f"Average sleep duration: {sleep_value} {sleep_unit}")
            if health_data.get('weight') and health_data['weight'].get('value'):
                weight_value = health_data['weight']['value']
                weight_unit = health_data['weight'].get('unit', 'kg')
                user_profile_parts.append(f"Weight: {weight_value} {weight_unit}")
            if health_data.get('blood_glucose') and health_data['blood_glucose'].get('value'):
                bg_value = health_data['blood_glucose']['value']
                bg_unit = health_data['blood_glucose'].get('unit', 'mg/dL')
                user_profile_parts.append(f"Blood Glucose: {bg_value} {bg_unit}")
            if health_data.get('oxygen_saturation') and health_data['oxygen_saturation'].get('value'):
                o2_value = health_data['oxygen_saturation']['value']
                o2_unit = health_data['oxygen_saturation'].get('unit', '%')
                user_profile_parts.append(f"Oxygen Saturation: {o2_value} {o2_unit}")
        
        user_profile_text = "\n".join(user_profile_parts)
        
        system_prompt = """You are an integrative medicine AI assistant trained to generate comprehensive personalized health profiles.

Generate a complete personalization profile that includes all 7 clusters in a structured format:

1. **User Goals & Therapeutic Focus** - Map user goals to therapeutic targets with a narrative summary and goal map table
2. **Lifestyle Pillars** - Actionable recommendations across 6 pillars (Sleep, Movement, Nutrition, Mindfulness, Social Connection, Environment)
3. **Nutrition Plan Tiering** - 3-phase plan (Reset, Rebuild, Sustain) with timeline and food category guidance
4. **Personalized Action Mapping** - Supplement recommendations with biomarker triggers in a structured table
5. **Supplement Plan** - System-based supplement recommendations organized by body systems
6. **Tiered Insight Delivery** - Progressive insights based on user readiness and engagement level
7. **Cross-Validated Logic** - Risk pattern identification and multimodal connections between symptoms, labs, and metrics

Format your response as structured text with clear section headers. Use tables where appropriate (e.g., for supplements, goals, nutrition phases).
Be warm, supportive, and use clear, accessible language. Connect recommendations to the user's specific health data."""

        user_prompt = f"""Generate a complete personalized health profile for this user:

{user_profile_text}

Please provide all 7 clusters in your response, formatted clearly with section headers."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
                try:
                    profile_data = json.loads(content)
                    return profile_data
                except json.JSONDecodeError:
                    pass
            
            profile_data = self._parse_text_response(content)
            
            return profile_data
            
        except Exception as e:
            logger.error(f"Error generating profile with LLM: {e}", exc_info=True)
            raise
    
    def _parse_text_response(self, content: str) -> Dict[str, Any]:
        """
        Parse text response from fine-tuned model into structured profile.
        Uses multiple strategies to handle various output formats robustly.
        """
        # Map section numbers to cluster names
        section_mapping = {
            1: "user_goals",
            2: "lifestyle_pillars", 
            3: "nutrition_plan_tiering",
            4: "personalized_action",
            5: "supplement_plan",
            6: "tiered_insights",
            7: "cross_validated_logic"
        }
        
        profile = {}
        
        # Try multiple regex patterns to find section headers (handle format variations)
        patterns_to_try = [
            # Pattern 1: "SECTION 1:", "Section 1:", "section 1:" with flexible separators
            r'(?:^|\n)(?:SECTION\s+(\d+)[\s:\-]+([^\n]+))',
            # Pattern 2: "1.", "1)", "1:", "1 -" 
            r'(?:^|\n)(\d+)[\.\)\:\-]\s+([^\n]+)',
            # Pattern 3: Markdown headers "### 1.", "## Section 1"
            r'(?:^|\n)#{1,6}\s*(?:SECTION\s+)?(\d+)[\.\:\s]+([^\n]+)',
            # Pattern 4: More relaxed "Section" with optional word variations
            r'(?:^|\n)(?:Section|Cluster|Part)\s+(\d+)[:\s]+([^\n]+)',
        ]
        
        matches = None
        matched_pattern_idx = -1
        
        for idx, pattern_str in enumerate(patterns_to_try):
            try:
                section_pattern = re.compile(pattern_str, re.MULTILINE | re.IGNORECASE)
                temp_matches = list(section_pattern.finditer(content))
                
                # Filter to only valid section numbers (1-7)
                valid_matches = []
                for m in temp_matches:
                    try:
                        num = int(m.group(1))
                        if 1 <= num <= 7:
                            valid_matches.append(m)
                    except (ValueError, IndexError):
                        continue
                
                # Use this pattern if we found at least 5 valid sections
                if len(valid_matches) >= 5:
                    matches = valid_matches
                    matched_pattern_idx = idx
                    logger.info(f"✅ Found {len(matches)} section headers using pattern {idx + 1}")
                    break
            except re.error as e:
                logger.warning(f"Regex pattern {idx + 1} failed: {e}")
                continue
        
        # Log matched sections for debugging
        if matches:
            logger.debug(f"Matched sections using pattern {matched_pattern_idx + 1}:")
            for m in matches[:10]:
                try:
                    section_num = int(m.group(1))
                    title = (m.group(2) or "").strip()
                    logger.debug(f"  Section {section_num}: {title[:60]}")
                except (ValueError, IndexError):
                    continue
        
        # Extract sections based on matched patterns
        if matches:
            for i, match in enumerate(matches):
                try:
                    section_num = int(match.group(1))
                    
                    if section_num not in section_mapping:
                        logger.debug(f"Skipping section {section_num} (not in mapping)")
                        continue
                    
                    cluster_name = section_mapping[section_num]
                    
                    # Start extraction after the header line
                    section_start = match.end()
                    
                    # Find end (next section or end of content)
                    if i + 1 < len(matches):
                        section_end = matches[i + 1].start()
                    else:
                        section_end = len(content)
                    
                    # Extract raw content
                    section_content = content[section_start:section_end].strip()
                    
                    # Clean content: remove separator lines and excessive whitespace
                    lines = section_content.split('\n')
                    cleaned_lines = []
                    
                    for line in lines:
                        stripped = line.strip()
                        # Skip separator lines (---, ===, ___, ***)
                        if re.match(r'^[\-=_\*]{3,}$', stripped):
                            continue
                        # Skip lines that are just numbers or section markers
                        if re.match(r'^(?:Section|SECTION)?\s*\d+[\.\)\:]?\s*$', stripped, re.IGNORECASE):
                            continue
                        cleaned_lines.append(line)
                    
                    section_content = '\n'.join(cleaned_lines).strip()
                    
                    # Store if non-empty
                    if section_content:
                        profile[cluster_name] = section_content
                        logger.debug(f"✓ Extracted {cluster_name}: {len(section_content)} chars")
                    else:
                        logger.warning(f"Section {section_num} ({cluster_name}) is empty after cleaning")
                    
                except (ValueError, IndexError, AttributeError) as e:
                    logger.warning(f"Error parsing match {i}: {e}")
                    continue
        
        # Fallback Strategy 1: Try separator-based splitting
        if len(profile) < 5:
            logger.warning(f"Only found {len(profile)}/7 sections with patterns, trying separator splitting")
            sections = re.split(r'\n\s*[\-=]{3,}\s*\n', content)
            
            if len(sections) >= 7:
                for i, section_text in enumerate(sections[:7], 1):
                    cluster_name = section_mapping.get(i)
                    if cluster_name and cluster_name not in profile:
                        cleaned = section_text.strip()
                        if cleaned and len(cleaned) > 20:  # Minimum content check
                            profile[cluster_name] = cleaned
                            logger.info(f"✓ Extracted {cluster_name} via separator splitting")
        
        # Fallback Strategy 2: Keyword-based extraction
        if len(profile) < 5:
            logger.warning(f"Only found {len(profile)}/7 sections, trying keyword-based extraction")
            section_keywords = {
                "user_goals": ["user goal", "therapeutic focus", "narrative summary"],
                "lifestyle_pillars": ["lifestyle pillar", "sleep", "movement", "stress"],
                "nutrition_plan_tiering": ["nutrition plan", "phase 1", "phase 2"],
                "personalized_action": ["personalized action", "supplement recommendation"],
                "supplement_plan": ["supplement plan", "cardiovascular", "metabolic"],
                "tiered_insights": ["tiered insight", "level 1", "level 2"],
                "cross_validated_logic": ["cross validated", "identified pattern", "synergistic"]
            }
            
            keyword_profile = self._extract_by_keywords(content, section_keywords)
            for key, value in keyword_profile.items():
                if key not in profile and value:
                    profile[key] = value
                    logger.info(f"✓ Extracted {key} via keyword matching")
        
        # Final extraction summary
        logger.info(f"📊 Extraction complete: {len(profile)}/7 sections found")
        for cluster_name in section_mapping.values():
            if cluster_name in profile:
                char_count = len(profile[cluster_name])
                logger.debug(f"  ✓ {cluster_name}: {char_count} chars")
            else:
                logger.warning(f"  ✗ {cluster_name}: MISSING")
        
        # Validate all required clusters are present
        missing_clusters = [name for name in section_mapping.values() if not profile.get(name)]
        if missing_clusters:
            logger.error(f"❌ Validation failed. Missing clusters: {missing_clusters}")
            logger.error(f"Parsed clusters: {list(profile.keys())}")
            logger.error(f"Raw content preview:\n{content[:2000]}")
            raise ValueError(f"Model response missing required clusters: {missing_clusters}")
        
        return profile
    
    def _extract_by_keywords(self, content: str, section_keywords: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Fallback: Extract sections by looking for known keywords.
        """
        profile = {}
        content_lower = content.lower()
        
        # Find positions of keywords for each section
        section_positions = {}
        for cluster_name, keywords in section_keywords.items():
            for keyword in keywords:
                pos = content_lower.find(keyword.lower())
                if pos != -1:
                    section_positions[cluster_name] = pos
                    break
        
        if not section_positions:
            return profile
        
        # Sort by position
        sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])
        
        # Extract content between positions
        for i, (cluster_name, start_pos) in enumerate(sorted_sections):
            if i + 1 < len(sorted_sections):
                end_pos = sorted_sections[i + 1][1]
            else:
                end_pos = len(content)
            
            section_content = content[start_pos:end_pos].strip()
            
            # Skip the first line (keyword line)
            lines = section_content.split('\n')
            if len(lines) > 1:
                section_content = '\n'.join(lines[1:]).strip()
            
            # Only store if meaningful content
            if section_content and len(section_content) > 30:
                profile[cluster_name] = section_content
        
        return profile
    
    def _validate_profile(self, profile_data: Dict[str, Any]) -> bool:
        """
        Validate that all required clusters are present in the profile.
        """
        REQUIRED_CLUSTERS = [
            "user_goals",
            "lifestyle_pillars",
            "nutrition_plan_tiering",
            "personalized_action",
            "supplement_plan",
            "tiered_insights",
            "cross_validated_logic"
        ]
        
        missing = [cluster for cluster in REQUIRED_CLUSTERS if not profile_data.get(cluster)]
        if missing:
            logger.error(f"Profile missing clusters: {missing}")
            return False
        return True
    
    async def _save_profile(self, user_email: str, profile: Dict[str, Any]):
        """Save personalization profile to user document."""
        profile_to_save = profile.copy()
        if "generated_at" in profile_to_save:
            profile_to_save["generated_at"] = profile_to_save["generated_at"].isoformat()
        
        result = await self.user_collection.update_one(
            {"email": user_email},
            {
                "$set": {
                    "personalization_profile": profile_to_save,
                    "profile_updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        if result.modified_count == 0:
            logger.warning(f"Profile not saved for {user_email}")
    
    def format_profile_for_chat(self, profile: Dict[str, Any]) -> str:
        """
        Format personalization profile for chat context.
        
        """
        if not profile or "data" not in profile:
            return ""
        
        data = profile["data"]
        summary_parts = []
        
        if "user_goals" in data and data["user_goals"]:
            summary_parts.append(f"USER GOALS & THERAPEUTIC FOCUS:\n{str(data['user_goals'])}")
        
        if "lifestyle_pillars" in data and data["lifestyle_pillars"]:
            summary_parts.append(f"LIFESTYLE PILLARS:\n{str(data['lifestyle_pillars'])}")
        
        if "nutrition_plan_tiering" in data and data["nutrition_plan_tiering"]:
            summary_parts.append(f"NUTRITION PLAN TIERING:\n{str(data['nutrition_plan_tiering'])}")
        
        if "personalized_action" in data and data["personalized_action"]:
            summary_parts.append(f"PERSONALIZED ACTION MAPPING:\n{str(data['personalized_action'])}")
        
        if "supplement_plan" in data and data["supplement_plan"]:
            summary_parts.append(f"SUPPLEMENT PLAN:\n{str(data['supplement_plan'])}")
        
        if "tiered_insights" in data and data["tiered_insights"]:
            summary_parts.append(f"TIERED INSIGHT DELIVERY:\n{str(data['tiered_insights'])}")
        
        if "cross_validated_logic" in data and data["cross_validated_logic"]:
            summary_parts.append(f"CROSS-VALIDATED LOGIC & RISK PATTERNS:\n{str(data['cross_validated_logic'])}")
        
        if not summary_parts:
            return ""
        
        return "\n\n".join(summary_parts)


_personalization_service_instance = None


def get_personalization_profile_service() -> PersonalizationProfileService:
    """Get singleton instance of PersonalizationProfileService."""
    global _personalization_service_instance
    if _personalization_service_instance is None:
        _personalization_service_instance = PersonalizationProfileService()
    return _personalization_service_instance


def reset_personalization_service():
    """Reset the singleton instance - useful for testing or reloading"""
    global _personalization_service_instance
    _personalization_service_instance = None
