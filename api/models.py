"""
Pydantic models for request/response schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in the conversation"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    user_id: str = Field(..., description="Unique user identifier")
    message: str = Field(..., description="User's message")
    conversation_history: Optional[List[Message]] = Field(
        default=None, 
        description="Recent conversation history (optional, will use Redis cache if not provided)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "message": "I'm so burnt out today. Did 6 hours of LeetCode and still feel stuck.",
                "conversation_history": None
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Nudge's response")
    user_id: str
    memories_used: int = Field(..., description="Number of memories retrieved for context")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MemoryEntry(BaseModel):
    """A memory entry to store"""
    user_id: str
    content: str
    memory_type: str = Field(default="conversation", description="Type: conversation, goal, win, struggle, etc.")
    metadata: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StoreMemoryRequest(BaseModel):
    """Request to store a new memory"""
    user_id: str
    content: str
    memory_type: str = "conversation"
    metadata: Optional[dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "content": "User shipped their first Redis caching feature to production",
                "memory_type": "win",
                "metadata": {"project": "PMArchitect", "date": "2025-12-01"}
            }
        }


class UserProfile(BaseModel):
    """User profile with goals and context"""
    user_id: str
    name: Optional[str] = None
    goals: List[str] = []
    current_projects: List[str] = []
    job_status: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "name": "Rahul",
                "goals": ["Crack FAANG", "Build PMArchitect side project"],
                "current_projects": ["PMArchitect", "LeetCode grind"],
                "job_status": "Preparing for placements"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    llm_provider: str
    memory_status: str
    version: str = "1.0.0"


# =============================================
# IMPROVED ADVISOR NUDGE MODELS
# =============================================

class ImprovedNudgeRequest(BaseModel):
    """Request for personalized nudge with McKenna-style coaching"""
    user_id: str = Field(..., description="Unique user identifier")
    dream: str = Field(..., description="User's bold dream/goal")
    progress: dict = Field(
        default_factory=dict,
        description="Progress metrics: days_active, wins, struggles"
    )
    personality: dict = Field(
        default_factory=dict,
        description="Personality traits: energy_level, preferred_style"
    )
    history: Optional[List[Message]] = Field(
        default=None,
        description="Optional conversation history"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "dream": "Become a senior AI engineer at FAANG by 2026",
                "progress": {
                    "days_active": 45,
                    "wins": 12,
                    "struggles": ["procrastination", "imposter syndrome"]
                },
                "personality": {
                    "energy_level": "low",
                    "preferred_style": "gentle"
                },
                "history": None
            }
        }


class VisualizationStep(BaseModel):
    """A single step in a guided visualization"""
    text: str = Field(..., description="Visualization text for this step")
    duration_seconds: int = Field(default=20, description="Duration in seconds")


class ImprovedNudgeResponse(BaseModel):
    """Response with personalized nudge and visualization"""
    nudge: str = Field(..., description="Actionable micro-step (≤10 min)")
    visualization: dict = Field(
        ...,
        description="Guided visualization session with steps"
    )
    personality_insights: dict = Field(
        default_factory=dict,
        description="Extracted personality traits from memory"
    )
    tts_ready: bool = Field(
        default=False,
        description="Whether TTS audio generation is available"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "nudge": "Open LeetCode and solve problem #2389. Focus on the pattern, not perfection. Done? Yes/No",
                "visualization": {
                    "title": "Future Self: The Calm Engineer",
                    "full_text": "Close your eyes. Feel your breath...",
                    "steps": [
                        "Close your eyes. Feel your breath slowing...",
                        "See yourself in 2026, shipping code daily...",
                        "Hear the sound of your keyboard..."
                    ],
                    "duration_seconds": 90
                },
                "personality_insights": {
                    "energy": "low",
                    "style": "gentle",
                    "tends_toward": "overwhelm"
                },
                "tts_ready": False
            }
        }
