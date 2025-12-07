from fastapi import APIRouter, HTTPException, Body, Query, Depends
from app.services.backend_services.review_service import get_review_service
from app.schemas.backend.review import ReviewCreate, Review
from app.schemas.utils import HttpResponse
from typing import List
from app.core.security import get_current_user, CurrentUser

review_router = APIRouter()
review_service = get_review_service()

@review_router.post("/api/review/submit", response_model=HttpResponse)
async def submit_review(
    review_data: ReviewCreate = Body(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Submit customer feedback/review.
    
    - **rating**: Rating out of 5 (1-5)
    - **feedback**: Optional feedback text (encrypted for privacy)
    """
    try:
        # Set user_email from token
        review_data.user_email = current_user.email
        review = await review_service.create_review(review_data)
        return HttpResponse(
            success=True,
            message="Review submitted successfully",
            data={"review": review.model_dump()}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit review: {str(e)}")

@review_router.get("/api/review/user", response_model=HttpResponse)
async def get_user_reviews(current_user: CurrentUser = Depends(get_current_user)):
    """
    Get all reviews submitted by the current user.
    """
    try:
        reviews = await review_service.get_reviews_by_user(current_user.email)
        return HttpResponse(
            success=True,
            message="Reviews retrieved successfully",
            data={"reviews": [review.model_dump() for review in reviews]}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve reviews: {str(e)}")

