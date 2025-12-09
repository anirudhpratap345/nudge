"""
Configuration management for Nudge Coach API
Updated for Vercel - uses environment variables directly (no .env file)
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Settings
    app_name: str = "Nudge Coach API"
    debug: bool = False
    api_prefix: str = "/api/v1"
    
    # LLM Provider: "groq" or "local"
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    
    # Groq Settings (free tier: 500+ tok/s)
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    # Local/Self-hosted Settings (for fine-tuned model)
    local_model_path: str = os.getenv("LOCAL_MODEL_PATH", "../nudge-lora-v2/nudge-lora-v2")
    local_base_model: str = os.getenv("LOCAL_BASE_MODEL", "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
    
    # Embedding Settings (using lighter model for better compatibility)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    # ChromaDB Settings
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma_db")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "nudge_memories")
    
    # Redis Settings (for recent message cache)
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    redis_ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    redis_cache_size: int = int(os.getenv("REDIS_CACHE_SIZE", "10"))
    redis_enabled: bool = os.getenv("REDIS_ENABLED", "true").lower() == "true"
    
    # Memory Retrieval Settings
    memory_top_k: int = int(os.getenv("MEMORY_TOP_K", "8"))
    
    # Generation Settings
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    
    class Config:
        # Vercel uses environment variables directly - no .env file needed
        env_file = None
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

