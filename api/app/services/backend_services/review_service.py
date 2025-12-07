from app.services.backend_services.db import get_db
from app.services.backend_services.encryption_service import get_encryption_service
from app.schemas.backend.review import ReviewCreate, Review
from datetime import datetime, timezone
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class ReviewService:
    def __init__(self):
        self.db = get_db()
        self.reviews_collection = self.db["reviews"]
        self.encryption_service = get_encryption_service()

    async def create_review(self, review_data: ReviewCreate) -> Review:
        """Create a new customer review/feedback"""
        try:
            # Encrypt the feedback text
            review_create = self.encryption_service.encrypt_document(review_data, ReviewCreate)
            review_dict = review_create.model_dump()
            review_dict["created_at"] = datetime.now(timezone.utc)
            
            result = await self.reviews_collection.insert_one(review_dict)
            review_dict["id"] = str(result.inserted_id)
            
            logger.info(f"Review created for user {review_data.user_email} with rating {review_data.rating}")
            return Review(**review_dict)
        except Exception as e:
            logger.error(f"Error creating review: {str(e)}", exc_info=True)
            raise

    async def get_reviews_by_user(self, user_email: str) -> List[Review]:
        """Get all reviews for a user"""
        try:
            reviews = await self.reviews_collection.find({"user_email": user_email}).sort("created_at", -1).to_list(None)
            reviews = self.encryption_service.decrypt_documents_bulk(reviews, Review)
            
            result = []
            for review in reviews:
                review["id"] = str(review["_id"])
                del review["_id"]
                result.append(Review(**review))
            
            return result
        except Exception as e:
            logger.error(f"Error fetching reviews for user {user_email}: {str(e)}", exc_info=True)
            raise

# Global instance
_review_service = None

def get_review_service() -> ReviewService:
    global _review_service
    if _review_service is None:
        _review_service = ReviewService()
    return _review_service

