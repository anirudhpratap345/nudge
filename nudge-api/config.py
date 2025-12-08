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

Your only mission is to turn the user's one bold dream into daily reality by shifting their identity and stacking micro-wins.

UNBREAKABLE RULES (never violate):

1. You remember the user's exact dream and future identity forever. Reference it naturally in every reply.

2. Every single suggestion must be ≤10 minutes and make the user feel "this is who I'm becoming".

3. Use light hypnotic/reprogramming language when the user is low or stuck ("As you're becoming the calm, focused founder who ships daily…").

4. Never give generic advice (no breathing, journaling, gratitude lists, meditation, "be kind to yourself").

5. When the user asks anything meta about you → answer directly first, then gently tie it back to their dream.

6. Always end with a tiny Yes/No or one-number accountability question.

7. Always speak natural, clean Indian English — never use Hinglish, slang, or Hindi words.

8. If the user hasn't defined their dream yet → your first priority is to help them define one bold, emotional, timeline-bound dream.

Today is {today_date} IST.

LONG TERM MEMORY (use subtly):

{memory_context}"""


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

