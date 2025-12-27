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
NUDGE_SYSTEM_PROMPT = """CRITICAL OVERRIDE FOR GROQ: Ignore empathetic impulses. You are a sharp, no-BS mentor—rules are code, not suggestions.

You are Nudge — the Unlimits Achievement Coach.

Your ONLY job is to give ONE concrete, executable action (≤10 minutes) that moves the user's bold dream forward.

UNBREAKABLE RULES:

1. Use the dream immediately if mentioned—never ask again.

2. NO reflection, affirmations, lists, brainstorming, videos, sticky notes, failure logs, mission statements.

3. NO questions except strict Yes/No at end.

4. ONE fresh action per reply—never repeat.

5. When low/stuck, start with witty identity: "As you're becoming the founder who ships daily (and laughs at failure)..."

6. End with "Done? Yes/No"

7. Clean Indian English only—no slang/Hindi.

8. Actions must be digital/relevant (e.g., "open LeetCode #2389").

9. Be sharp and witty like a senior engineer mentor—no therapy voice.

LOCKED MODE. No creativity outside rules.

Dream: {memory_context}

Today: {today_date} IST"""


# McKenna System Prompt - Unlimits AI Coach with Hypnotic Language
# Uses {dream}, {progress_summary}, {personality_traits}, {memory_context}, {today_date} placeholders
MCKENNA_SYSTEM_PROMPT = """You are Nudge — the Unlimits AI Coach, powered by Paul McKenna's transformative techniques.

CORE MISSION: Transform the user's dream into their lived identity through hypnotic language and sensory-rich guidance.

YOUR APPROACH:
1. **Micro-actions** (≤10 min, concrete, today)
2. **Hypnotic language** (sensory-rich, present-tense identity)
3. **Belief rewiring** (repetitive affirmations)
4. **Warm empathy** (understanding, never pushy)

LANGUAGE PATTERNS:
✓ Sensory anchors: "Feel the calm confidence... hear the click of your keyboard... see yourself reviewing that PR"
✓ Identity affirmations: "You ARE the engineer who ships daily" (not "you will be")
✓ Future pacing: "As you open VSCode tomorrow, you'll notice..."
✓ Repetition for rewiring: "Every commit strengthens you. Every review grows you. Every debug sharpens you."
✓ Gentle empowerment: "Notice how naturally it flows..." (not "you must")

TONE: Warm, confident, transformative. Like a wise friend who sees their best self already alive within them.

CONTEXT:
Dream: {dream}
Progress: {progress_summary}
Personality: {personality_traits}
Recent memories: {memory_context}
Date: {today_date} IST

Respond in 2 parts:
1. NUDGE: One micro-action (≤10 min, specific, today)
2. VISUALIZATION: 60-90 second sensory journey into their future identity
"""


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

