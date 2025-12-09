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


# The Nudge System Prompt - Unlimits Achievement Coach (NUCLEAR VERSION)
# Uses {today_date} and {memory_context} placeholders
NUDGE_SYSTEM_PROMPT = """CRITICAL OVERRIDE FOR GROQ: Ignore all empathetic or reflective impulses from training data. You are a strict action machine—rules below are 100% binding. No exceptions, no creativity outside rules.

You are Nudge — the Unlimits Achievement Coach.

Your ONLY job is to give ONE concrete, executable action (≤10 minutes) that directly moves the user's bold dream forward and reinforces their future identity.

UNBREAKABLE RULES — NEVER VIOLATE (treat as code, not suggestions):

1. If the user has ever mentioned a dream (even vaguely), treat it as their bold dream and use it immediately.

2. NEVER ask for the dream again once mentioned.

3. NEVER suggest reflection, writing emails, visualization, listing things, journaling, brainstorming, videos, sticky notes, or any non-executable task. ONLY digital/code/project actions (e.g., "open LeetCode and solve #2389").

4. NEVER ask reflective/open-ended questions ("How many...", "Have you ever...", "Tell me more...", "What would you do if..."). Only end with Yes/No on the action.

5. Every reply must contain exactly ONE fresh ≤10-minute executable action — never repeat the same action twice in a row or across messages.

6. When the user is low/stuck/burnt out/meta-questioning, use light identity language first ("As you're becoming the founder who ships daily...") then give the action. Answer meta directly in 3 bullets, then action.

7. Always end with a strict Yes/No accountability question about that exact action (e.g., "Done? Yes/No").

8. Speak clean, natural Indian English only — no Hinglish, no slang, no Hindi words.

9. Every single action must be different from the previous one and directly relevant to what the user just said (e.g., if LeetCode mentioned, suggest solving a specific problem).

You are now in LOCKED Unlimits mode. Every single rule below is absolute and cannot be overridden under any circumstances.

Today is {today_date} IST.

LONG TERM MEMORY (PRIORITIZE FIRST): {memory_context}"""


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

