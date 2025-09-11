from pydantic import BaseModel


class AIResponse(BaseModel):
    message: str
    error: str = ""
