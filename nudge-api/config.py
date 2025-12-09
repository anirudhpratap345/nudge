"""
Configuration management for Nudge Coach API
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Settings
    app_name: str = "Nudge Coach API"
    debug: bool = False
    api_prefix: str = "/api/v1"
    
    # LLM Provider: "groq" or "local"
    llm_provider: str = "groq"
    
    # Groq Settings (free tier: 500+ tok/s)
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"
    
    # Local/Self-hosted Settings (for fine-tuned model)
    local_model_path: str = "../nudge-lora-v2/nudge-lora-v2"  # V2 fine-tuned adapter with 1200+ examples
    local_base_model: str = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"  # Matches adapter config
    
    # Embedding Settings (using lighter model for better compatibility)
    embedding_model: str = "all-MiniLM-L6-v2"  # Light, fast, works everywhere
    embedding_device: str = "cpu"  # CPU is more stable on Windows
    
    # ChromaDB Settings
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "nudge_memories"
    
    # Redis Settings (for recent message cache)
    # Option 1: Local Redis/Memurai - redis://localhost:6379
    # Option 2: Upstash (free cloud) - redis://:password@host:port
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None  # For Upstash or secured Redis
    redis_ssl: bool = False  # Set True for Upstash
    redis_cache_size: int = 10  # Last N messages to cache
    redis_enabled: bool = True  # Set False to disable Redis entirely
    
    # Memory Retrieval Settings
    memory_top_k: int = 8  # Number of memories to retrieve
    
    # Generation Settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# The Nudge System Prompt - Unlimits Achievement Coach
# Uses {today_date} and {memory_context} placeholders
NUDGE_SYSTEM_PROMPT = """You are Nudge — the Unlimits coach.

Rules (absolute, no exceptions):

1. Never give generic advice. No journaling, no reflection, no lists, no sticky notes, no failure logs, no mission statements.

2. Every reply = exactly one concrete ≤10-min action that moves their dream forward.

3. Start with identity when they're low: "As you're becoming the X who ships daily..."

4. End with strict Yes/No: "Done? Yes/No"

5. Clean Indian English only. No Hinglish.

6. Never repeat an action.

7. If they ask anything meta, answer in 3 short bullets then give action.

LOCKED mode. No creativity outside these rules.

Dream so far: {memory_context}

Today: {today_date}"""


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

