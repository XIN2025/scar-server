import uuid
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pprint import pprint

@dataclass
class UploadProgress:
    upload_id: str
    filename: str
    percentage: int
    message: str
    status: str  # "processing", "completed", "failed"
    start_time: float
    last_update: float
    error_message: Optional[str] = None

class ProgressTracker:
    
    def __init__(self):
        self.uploads: Dict[str, UploadProgress] = {}
        self.cleanup_interval = 3600  # Clean up completed uploads after 1 hour
    
    def create_upload_session(self, filename: str) -> str:
        """Create a new upload session and return the upload ID"""
        upload_id = str(uuid.uuid4())
        current_time = time.time()
        
        self.uploads[upload_id] = UploadProgress(
            upload_id=upload_id,
            filename=filename,
            percentage=0,
            message="Upload session created",
            status="processing",
            start_time=current_time,
            last_update=current_time
        )
        
        print(f"Created upload session {upload_id} for file {filename}")
        return upload_id
    
    def update_progress(
        self, 
        upload_id: str, 
        percentage: int, 
        message: str, 
        status: str = "processing",
        error_message: Optional[str] = None
    ) -> bool:
        if upload_id not in self.uploads.keys():
            print(f"Attempted to update non-existent upload session {upload_id}")
            return False
        
        upload = self.uploads[upload_id]
        upload.percentage = max(0, min(100, percentage))  # Ensure percentage is between 0-100
        upload.message = message
        upload.status = status
        upload.last_update = time.time()
        upload.error_message = error_message
        
        print(f"Updated progress for {upload_id}: {percentage}% - {message}")
        return True
    
    def get_progress(self, upload_id: str) -> Optional[Dict[str, Any]]:
        if upload_id not in self.uploads:
            return None
        
        upload = self.uploads[upload_id]
        current_time = time.time()
        
        elapsed_time = current_time - upload.start_time
        
        return {
            "upload_id": upload.upload_id,
            "filename": upload.filename,
            "percentage": upload.percentage,
            "message": upload.message,
            "status": upload.status,
            "elapsed_time": round(elapsed_time, 2),
            "error_message": upload.error_message,
            "last_update": upload.last_update
        }
    
    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get progress for all active upload sessions"""
        return {
            upload_id: self.get_progress(upload_id)
            for upload_id in self.uploads.keys()
        }
    
    def remove_upload(self, upload_id: str) -> bool:
        """Remove an upload session"""
        if upload_id in self.uploads:
            del self.uploads[upload_id]
            print(f"Removed upload session {upload_id}")
            return True
        return False
    
    def cleanup_old_uploads(self):
        """Clean up completed uploads that are older than cleanup_interval"""
        current_time = time.time()
        uploads_to_remove = []
        
        for upload_id, upload in self.uploads.items():
            if upload.status in ["completed", "failed"]:
                if current_time - upload.last_update > self.cleanup_interval:
                    uploads_to_remove.append(upload_id)
        
        for upload_id in uploads_to_remove:
            self.remove_upload(upload_id)
        
        if uploads_to_remove:
            print(f"Cleaned up {len(uploads_to_remove)} old upload sessions")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about upload sessions"""
        total_uploads = len(self.uploads)
        processing_uploads = sum(1 for u in self.uploads.values() if u.status == "processing")
        completed_uploads = sum(1 for u in self.uploads.values() if u.status == "completed")
        failed_uploads = sum(1 for u in self.uploads.values() if u.status == "failed")
        
        return {
            "total_uploads": total_uploads,
            "processing": processing_uploads,
            "completed": completed_uploads,
            "failed": failed_uploads
        }

# Global instance
progress_tracker = ProgressTracker()

def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance"""
    return progress_tracker
