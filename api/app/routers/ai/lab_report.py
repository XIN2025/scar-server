from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends
from pydantic import EmailStr
from typing import List
import logging
import asyncio
from urllib.parse import unquote

from app.services.ai_services.lab_report_service import get_lab_report_service
from app.services.backend_services.health_score_service import get_health_score_service
from app.services.backend_services.progress_tracker import get_progress_tracker
from app.schemas.ai.lab_report import LabReportResponse, LabReportSummary, LabReportUploadResponse
from app.schemas.backend.documents import DocumentType
from app.core.security import get_current_user, CurrentUser

logger = logging.getLogger(__name__)

lab_report_router = APIRouter(prefix="/lab-reports", tags=["lab-reports"])

@lab_report_router.post("/upload", response_model=LabReportUploadResponse)
async def upload_lab_report(
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Upload a PDF lab report and extract structured data using AI
    
    - **file**: PDF file containing the lab report
    - **user_email**: Email of the user who owns this lab report
    
    Returns the extracted lab report data with structured test results
    """
    try:
        # URL decode filename (iOS may send URL-encoded filenames)
        filename = unquote(file.filename) if file.filename else None
        
        # Validate file type
        if not filename:
            logger.error("No file provided")
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not filename.lower().endswith('.pdf'):
            logger.error("Only PDF files are supported for lab reports")
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are supported for lab reports"
            )
        
        # Read file content FIRST (file.size is unreliable on iOS - often None)
        file_content = await file.read()
        content_size = len(file_content)
        logger.info(f"File size: {content_size} bytes")
        
        if content_size == 0:
            logger.error("Uploaded file is empty")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Check file size (25MB limit)
        max_size = 25 * 1024 * 1024  # 25MB in bytes
        if content_size > max_size:
            logger.error(f"File too large: {content_size} bytes")
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size allowed is {max_size // (1024*1024)}MB"
            )
        
        logger.info(f"Processing lab report PDF: {filename} for user: {current_user.email}")
        
        # Initialize services
        lab_report_service = get_lab_report_service()
        progress_tracker = get_progress_tracker()
        
        # Create upload session immediately for progress tracking
        upload_id = progress_tracker.create_upload_session(filename)
        progress_tracker.update_progress(upload_id, 5, "File uploaded, starting processing...")
        
        # Process in background - return immediately for better UX
        async def process_in_background():
            try:
                lab_report = await lab_report_service.process_lab_report_pdf(
                    file_content, filename, current_user.email, DocumentType.LAB_REPORT, upload_id
                )
                
                # Trigger health score recalculation in background (non-blocking)
                try:
                    health_score_service = get_health_score_service()
                    asyncio.create_task(
                            health_score_service._background_recalculate_health_score(current_user.email)
                    )
                except Exception as score_error:
                    logger.warning(
                        f"Failed to trigger health score recalculation for {user_email}: {score_error}"
                    )
                
                logger.info(f"Background processing completed for upload_id: {upload_id}")
            except Exception as bg_error:
                logger.error(f"Background processing failed for upload_id {upload_id}: {bg_error}")
                progress_tracker.update_progress(
                    upload_id,
                    100,
                    f"Processing failed: {str(bg_error)}",
                    "failed",
                    error_message=str(bg_error)
                )
        
        # Start background processing
        asyncio.create_task(process_in_background())
        
        # Return immediately with upload_id for progress tracking
        return LabReportUploadResponse(
            success=True,
            upload_id=upload_id,
            lab_report_id=None,
            message="Lab report upload accepted. Processing in background. Use upload_id to track progress.",
            test_title=None,
            test_description=None,
            properties_count=None,
            processing=True
        )
        
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error processing lab report upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process lab report: {str(e)}"
        )



@lab_report_router.get("/", response_model=List[LabReportSummary])
async def get_lab_reports(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get all lab reports for a user (summary view)
    
    Returns a list of lab report summaries with basic information
    """
    try:
        lab_report_service = get_lab_report_service()
        reports = await lab_report_service.get_lab_reports_by_user(current_user.email)
        
        return reports
        
    except Exception as e:
        logger.error(f"Error fetching lab reports for user {current_user.email}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch lab reports: {str(e)}"
        )

@lab_report_router.get("/{report_id}", response_model=LabReportResponse)
async def get_lab_report_by_id(
    report_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get detailed lab report by ID
    
    - **report_id**: Unique identifier of the lab report
    
    Returns detailed lab report with all test properties and values
    """
    try:
        lab_report_service = get_lab_report_service()
        report = await lab_report_service.get_lab_report_by_id(report_id, current_user.email)
        
        if not report:
            raise HTTPException(
                status_code=404,
                detail="Lab report not found or you don't have permission to access it"
            )
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching lab report {report_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch lab report: {str(e)}"
        )

@lab_report_router.delete("/{upload_id}")
async def delete_lab_report(
    upload_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Delete a lab report by upload_id
    
    - **upload_id**: Upload ID of the lab report to delete
    - **user_email**: Email of the user (for authorization - can only delete own reports)
    
    Returns success message if deletion was successful
    """
    try:
        from app.services.backend_services.health_score_service import get_health_score_service
        
        lab_report_service = get_lab_report_service()
        
        lab_report = await lab_report_service.collection.find_one(
            {"upload_id": upload_id}
        )
        
        if not lab_report:
            raise HTTPException(
                status_code=404,
                detail="Lab report not found for the given upload_id"
            )
        
        owner_email = lab_report.get("user_email")
        
        # Authorization: ensure the caller owns this upload_id
        if current_user.email != owner_email:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to delete this lab report"
            )
        
        success = await lab_report_service.delete_lab_report_by_upload_id(upload_id, current_user.email)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Lab report not found in database"
            )
        
        # Trigger health score recalculation after deletion
        try:
                health_score_service = get_health_score_service()
                asyncio.create_task(
                    health_score_service._background_recalculate_health_score(current_user.email)
                )
        except Exception as score_error:
                logger.warning(
                    f"Failed to trigger health score recalculation for {user_email}: {score_error}"
                )
        
        return {
            "success": True,
            "message": "Lab report and associated data deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting lab report with upload_id {upload_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete lab report: {str(e)}"
        )
