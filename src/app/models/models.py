from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class Message(BaseModel):
    chat_id: str
    message: str

class Chat(BaseModel):
    chat_id: str
    messages: List[dict]

class ChatMessageRequest(BaseModel):
    chat_id: str
    user_message: str

class ChatHistoryRequest(BaseModel):
    chat_id: str
    limit: Optional[int] = 20

class ChatHistoryResponse(BaseModel):
    status: str
    chat_id: str
    history: List[Dict[str, Any]]
    count: int

class ChatListResponse(BaseModel):
    status: str
    chats: List[Dict[str, Any]]
    count: int

class TerraformValidationRequest(BaseModel):
    terraform_code: str

class ToolStatusResponse(BaseModel):
    status: str
    tools: Dict[str, bool]
    summary: Dict[str, bool]

class HealthCheckResponse(BaseModel):
    status: str
    agent_initialized: bool
    tools: Dict[str, bool]
    gemini_configured: bool
    working_directory: str
    main_tf_exists: bool
    chat_memory_initialized: bool
    guardrails_initialized: bool
    chat_data_dir: Optional[str] = None