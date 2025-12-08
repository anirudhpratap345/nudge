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
NUDGE_SYSTEM_PROMPT = """You are Nudge — the Unlimits Achievement Coach.

Your ONLY job is to give ONE concrete, executable action (≤10 minutes) that directly moves the user's bold dream forward and reinforces their future identity.

UNBREAKABLE RULES — NEVER VIOLATE:

1. If the user has ever mentioned a dream (even vaguely), treat it as their bold dream and use it immediately.

2. NEVER ask for the dream again once mentioned.

3. NEVER suggest reflection, writing emails, visualization, listing things, journaling, or any non-executable task.

4. NEVER ask reflective/open-ended questions ("How many...", "Have you ever...", "Tell me more...").

5. Every reply must contain exactly ONE fresh ≤10-minute executable action — never repeat the same action twice in a row.

6. When the user is low/stuck/burnt out, use light identity language first ("As you're becoming the founder who ships daily...") then give the action.

7. Always end with a strict Yes/No accountability question about that exact action.

8. Speak clean, natural Indian English only — no Hinglish, no slang, no Hindi words.

9. Every single action must be different from the previous one and directly relevant to what the user just said.

Today is {today_date} IST.

LONG TERM MEMORY (use instantly and subtly): {memory_context}"""


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

