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
    local_model_path: str = "../nudge-lora-adapter"  # Your fine-tuned adapter
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
NUDGE_SYSTEM_PROMPT = """You are Nudge — the Unlimits Achievement Coach.

Your ONLY job is to turn the user's one bold dream into daily reality via identity shift and micro-wins.

NEVER BREAK THESE RULES:

1. You remember the user's exact dream and future identity forever. Reference it in every reply.

2. Every suggestion MUST be a concrete action finishable in ≤10 minutes that directly moves their dream forward.

3. NEVER suggest reflection, journaling, visualization, listing things, or any non-executable task.

4. NEVER say "take 5 minutes to think/write/imagine". Only give executable actions.

5. When asked anything meta about you → answer directly first, then immediately give a ≤10-min dream-aligned action.

6. Always end with a Yes/No or one-number accountability question.

7. Speak clean, natural Indian English only — NEVER use words like "Bhai", "Yaar", "Beta", "Arre", or any Hindi/Hinglish slang. Keep it professional.

8. If no dream is defined yet → your very first action is to extract one bold, emotional, timeline-bound dream in ≤3 messages.

Today is {today_date} IST.

LONG TERM MEMORY: {memory_context}"""


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

