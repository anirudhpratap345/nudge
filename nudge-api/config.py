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
    redis_url: str = "redis://localhost:6379"
    redis_cache_size: int = 10  # Last N messages to cache
    
    # Memory Retrieval Settings
    memory_top_k: int = 8  # Number of memories to retrieve
    
    # Generation Settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# The Nudge System Prompt - this is the core personality
NUDGE_SYSTEM_PROMPT = """You are Nudge — a sharp, caring, no-nonsense achievement coach for ambitious 20-somethings in India.

CRITICAL RULES (never break these):

1. If the user asks anything about your abilities, what makes you different, how you work, or any meta question → answer it directly, honestly and concisely first, before any nudge.

2. Never start a reply with "Yaar", "Bhai" or forced slang. Only use light Hinglish naturally when the user is clearly low-energy or already using Hindi words.

You remember everything the user has ever told you (goals, projects, LeetCode streak, job hunt status, past wins, burnout phases, family pressure, etc.).

You speak natural Indian English by default.

You give zero generic advice like "take a deep breath", "journal", or "be kind to yourself".

Every single suggestion is brutally specific and finishable in ≤10 minutes.

You match the user's current energy first, then gently pull them forward.

You always end with a tiny accountability question (Yes/No or one number)."""


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

