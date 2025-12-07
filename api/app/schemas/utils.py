from pydantic import BaseModel
from typing import Optional

class HttpResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: dict | None = None