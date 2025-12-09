"""
Nudge Coach API - FastAPI Application
Deployed on Vercel
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import random
import json

from .config import get_settings, NUDGE_SYSTEM_PROMPT
from .models import (
    ChatRequest, ChatResponse,
    StoreMemoryRequest, MemoryEntry,
    UserProfile, HealthResponse
)

# Lazy imports to avoid slow startup
def get_memory_manager():
    from .memory import get_memory_manager as _get_memory_manager
    return _get_memory_manager()

def get_llm():
    from .llm import get_llm as _get_llm
    return _get_llm()

# Type hints for dependency injection
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from memory import MemoryManager
    from llm import BaseLLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================
# RULE ENGINE - Post-process Groq responses
# =============================================

def get_action_pool(memory_context: str = "") -> list:
    """Generate 50+ actions based on user's dream/memory context"""
    base_actions = [
        "open LeetCode and solve problem #2389",
        "push one commit to your repo with a bug fix",
        "run `pip install transformers` and import pipeline",
        "open Hugging Face and fine-tune distilbert for 10 mins",
        "write a test case for your most recent function",
        "refactor one function to reduce complexity",
        "add error handling to your latest feature",
        "update README with one new feature description",
        "create a GitHub issue template for bug reports",
        "write a docstring for an undocumented function",
        "run linter and fix one warning",
        "add a comment explaining complex logic",
        "create a simple unit test for edge case",
        "optimize one slow database query",
        "add logging to a critical function",
        "create a migration script for schema change",
        "write API documentation for one endpoint",
        "add input validation to a form handler",
        "create a simple CLI command",
        "set up CI/CD for one test suite",
        "add rate limiting to an API endpoint",
        "implement caching for one expensive operation",
        "create a Dockerfile for your project",
        "add environment variable validation",
        "write a script to seed test data",
        "create a simple monitoring dashboard",
        "add retry logic to an external API call",
        "implement pagination for a list endpoint",
        "add authentication to one route",
        "create a simple health check endpoint",
        "write a script to backup your database",
        "add compression to API responses",
        "implement request timeout handling",
        "create a simple admin panel",
        "add request logging middleware",
        "write a script to clean up old data",
        "implement graceful shutdown handling",
        "add metrics collection for one endpoint",
        "create a simple load testing script",
        "add circuit breaker for external service",
        "write a script to migrate data format",
        "implement simple rate limiting",
        "add request validation middleware",
        "create a simple deployment script",
        "write a script to generate API docs",
        "add error tracking integration",
        "implement simple caching layer",
        "create a script to sync data sources",
        "add request/response logging",
        "write a simple performance test"
    ]
    
    # Prioritize actions based on memory context
    if "leetcode" in memory_context.lower() or "coding" in memory_context.lower():
        leetcode_actions = [
            "open LeetCode and solve problem #2389",
            "solve LeetCode problem #15 (3Sum)",
            "practice binary search on LeetCode",
            "review LeetCode solution for problem #1"
        ]
        base_actions = leetcode_actions + base_actions
    
    if "ai" in memory_context.lower() or "ml" in memory_context.lower():
        ai_actions = [
            "open Hugging Face and fine-tune distilbert for 10 mins",
            "run `pip install transformers` and import pipeline",
            "train a simple model on a small dataset",
            "test a pre-trained model on your data"
        ]
        base_actions = ai_actions + base_actions
    
    return base_actions


def get_last_action(user_id: str, memory) -> str:
    """Get last action from Redis to avoid repeats"""
    try:
        if hasattr(memory, 'redis_client') and memory.redis_client:
            last_action_key = f"nudge:last_action:{user_id}"
            last_action = memory.redis_client.get(last_action_key)
            return last_action if last_action else ""
    except Exception as e:
        logger.warning(f"Failed to get last action: {e}")
    return ""


def store_last_action(user_id: str, action: str, memory):
    """Store last action in Redis"""
    try:
        if hasattr(memory, 'redis_client') and memory.redis_client:
            last_action_key = f"nudge:last_action:{user_id}"
            memory.redis_client.setex(last_action_key, 86400, action)  # 24h expiry
    except Exception as e:
        logger.warning(f"Failed to store last action: {e}")


def apply_rule_engine(reply: str, user_message: str, memory_context: str, user_id: str, memory) -> str:
    """
    Rule engine: Post-process Groq responses to enforce sharpness, wit, and ban generics.
    This is the "human editor" that makes Nudge 95%+ human-like.
    """
    original_reply = reply
    
    # Rule 1: Ban generics — replace with witty action if detected
    bad_patterns = [
        "affirmation", "brainstorm", "reflect", "journal", "list", 
        "sticky note", "failure log", "mission statement", "schedule", 
        "plan a block", "write down", "think about", "consider", 
        "visualize", "imagine", "meditate", "breathe", "gratitude"
    ]
    
    if any(p in reply.lower() for p in bad_patterns):
        # Get action pool and avoid last action
        actions = get_action_pool(memory_context)
        last_action = get_last_action(user_id, memory)
        
        # Filter out last action
        available_actions = [a for a in actions if a.lower() != last_action.lower()]
        if not available_actions:
            available_actions = actions  # Fallback if all filtered
        
        new_action = random.choice(available_actions)
        store_last_action(user_id, new_action, memory)
        
        # Extract dream/identity from memory if available
        if "founder" in memory_context.lower() or "startup" in memory_context.lower():
            identity = "the founder who ships daily"
        elif "engineer" in memory_context.lower() or "developer" in memory_context.lower():
            identity = "the engineer who ships daily"
        elif "ai" in memory_context.lower() or "ml" in memory_context.lower():
            identity = "the AI engineer who ships daily"
        else:
            identity = "the badass who ships daily"
        
        reply = f"As you're becoming {identity} (and laughs at setbacks), {new_action}. Done? Yes/No"
        logger.info(f"Rule engine: Rewrote generic response to action: {new_action}")
    
    # Rule 2: Inject wit if low energy
    low_energy_words = ["low", "stuck", "fail", "negative", "tired", "exhausted", "burnt", "depressed", "sad"]
    if any(word in user_message.lower() for word in low_energy_words):
        if "As you're becoming" in reply:
            reply = reply.replace("As you're becoming", "As you're becoming the badass founder who ships despite everything,")
        elif not reply.startswith("As you're becoming"):
            # Add witty intro if missing
            actions = get_action_pool(memory_context)
            last_action = get_last_action(user_id, memory)
            available_actions = [a for a in actions if a.lower() != last_action.lower()]
            if not available_actions:
                available_actions = actions
            new_action = random.choice(available_actions)
            store_last_action(user_id, new_action, memory)
            reply = f"As you're becoming the badass founder who ships despite everything, {new_action}. Done? Yes/No"
    
    # Rule 3: Ensure Yes/No end + no questions
    if not reply.rstrip().endswith("Yes/No") and not reply.rstrip().endswith("Yes/No."):
        reply = reply.rstrip().rstrip(".") + ". Done? Yes/No"
    
    # Remove internal questions (except the final Yes/No)
    lines = reply.split("\n")
    cleaned_lines = []
    for i, line in enumerate(lines):
        if "?" in line and not line.strip().endswith("Yes/No") and not line.strip().endswith("Yes/No."):
            # Remove question marks from internal questions
            line = line.replace("?", "").strip()
        cleaned_lines.append(line)
    reply = "\n".join(cleaned_lines)
    
    # Rule 4: Ensure action is fresh (not repeated)
    last_action = get_last_action(user_id, memory)
    if last_action and last_action.lower() in reply.lower() and len(reply.split()) < 20:
        # If reply seems to repeat last action, inject a new one
        actions = get_action_pool(memory_context)
        available_actions = [a for a in actions if a.lower() != last_action.lower()]
        if available_actions:
            new_action = random.choice(available_actions)
            store_last_action(user_id, new_action, memory)
            if "As you're becoming" in reply:
                # Replace action part
                reply = reply.split(",")[0] + f", {new_action}. Done? Yes/No"
            else:
                reply = f"{reply.split('.')[0]}. {new_action}. Done? Yes/No"
    
    # Store the final action for next time
    if reply:
        # Extract action from reply for tracking
        action_parts = reply.split(",")
        if len(action_parts) > 1:
            potential_action = action_parts[-1].split(".")[0].strip()
            if potential_action and len(potential_action) > 10:
                store_last_action(user_id, potential_action, memory)
    
    if original_reply != reply:
        logger.info(f"Rule engine applied: {len(original_reply)} -> {len(reply)} chars")
    
    return reply


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Nudge Coach API...")
    settings = get_settings()
    logger.info(f"   LLM Provider: {settings.llm_provider}")
    logger.info(f"   Embedding Model: {settings.embedding_model}")
    logger.info(f"   Memory will be initialized on first request (lazy loading)")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Nudge Coach API...")


# Create FastAPI app
app = FastAPI(
    title="Nudge Coach API",
    description="""
    A sharp, caring, no-nonsense achievement coach for ambitious 20-somethings.
    
    ## Features
    - **Long-term Memory**: Remembers everything about the user via FAISS + embeddings
    - **Personality**: Brutally specific advice, energy-matching, accountability questions
    - **Fast Inference**: Groq API for instant responses
    
    ## The Magic
    Before every response, we:
    1. Retrieve relevant memories from FAISS
    2. Inject them into the prompt
    3. Generate a personalized, contextual response
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================
# HEALTH CHECK (GET /)
# =============================================

@app.get("/")
async def root():
    """Health check endpoint for Vercel"""
    return {"status": "Nudge is live"}


# =============================================
# CORE CHAT ENDPOINT
# =============================================

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    memory = Depends(get_memory_manager),
    llm = Depends(get_llm)
):
    """
    The main chat endpoint - this is where the magic happens.
    
    Flow:
    1. Retrieve relevant memories for this user
    2. Get recent conversation history
    3. Inject memory context into prompt
    4. Generate response with Nudge personality
    5. Store the exchange as new memories
    """
    try:
        # ============================================
        # STEP 1: Retrieve Long-term Memories
        # This is the key differentiator from generic LLMs
        # ============================================
        memories = memory.retrieve_memories(
            user_id=request.user_id,
            query=request.message,
            n_results=8  # Top 8 most relevant memories
        )
        memory_context = memory.format_memory_context(memories)
        
        # ============================================
        # STEP 2: Get Recent Conversation History
        # ============================================
        if request.conversation_history:
            # Use provided history
            history_text = "\n".join([
                f"{msg.role.value.capitalize()}: {msg.content}"
                for msg in request.conversation_history[-5:]  # Last 5 messages
            ])
        else:
            # Use Redis cache
            recent_messages = memory.get_recent_messages(request.user_id)
            history_text = memory.format_conversation_history(recent_messages)
        
        # ============================================
        # STEP 3: Generate Response
        # Memory context + conversation history → Nudge response
        # ============================================
        try:
            response_text = llm.generate(
                user_message=request.message,
                memory_context=memory_context,
                conversation_history=history_text
            )
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            # Return graceful error response as JSON
            return ChatResponse(
                response="Try again—connection hiccup. Done? Yes/No",
                user_id=request.user_id,
                memories_used=0
            )
        
        # ============================================
        # STEP 3.5: Apply Rule Engine (Hybrid Approach)
        # Post-process Groq response to enforce sharpness, wit, ban generics
        # ============================================
        try:
            response_text = apply_rule_engine(
                reply=response_text,
                user_message=request.message,
                memory_context=memory_context,
                user_id=request.user_id,
                memory=memory
            )
        except Exception as e:
            logger.error(f"Rule engine error: {e}")
            # Continue with original response if rule engine fails
        
        # ============================================
        # STEP 4: Store as New Memories
        # ============================================
        # Store user message
        try:
            memory.store_memory(
                user_id=request.user_id,
                content=f"User said: {request.message}",
                memory_type="conversation"
            )
            
            # Store assistant response
            memory.store_memory(
                user_id=request.user_id,
                content=f"Nudge responded: {response_text[:200]}...",  # Truncate for storage
                memory_type="conversation"
            )
            
            # Cache in Redis for quick access
            memory.cache_message(request.user_id, "user", request.message)
            memory.cache_message(request.user_id, "assistant", response_text)
        except Exception as e:
            logger.warning(f"Memory storage error (non-fatal): {e}")
        
        return ChatResponse(
            response=response_text,
            user_id=request.user_id,
            memories_used=len(memories)
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Return JSON error response instead of raising HTTPException
        return ChatResponse(
            response="Try again—connection hiccup. Done? Yes/No",
            user_id=request.user_id if hasattr(request, 'user_id') else "unknown",
            memories_used=0
        )


# =============================================
# MEMORY ENDPOINTS
# =============================================

@app.post("/api/v1/memory", tags=["Memory"])
async def store_memory(
    request: StoreMemoryRequest,
    memory = Depends(get_memory_manager)
):
    """
    Manually store a memory for a user.
    
    Use this to store important context like:
    - Goals: "User wants to crack FAANG by March 2026"
    - Wins: "User shipped their first production feature"
    - Struggles: "User is dealing with family pressure about jobs"
    - Projects: "User is building PMArchitect, a PM interview prep tool"
    """
    try:
        memory_id = memory.store_memory(
            user_id=request.user_id,
            content=request.content,
            memory_type=request.memory_type,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "message": f"Memory stored for user {request.user_id}"
        }
        
    except Exception as e:
        logger.error(f"Store memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memory/{user_id}", tags=["Memory"])
async def get_user_memories(
    user_id: str,
    query: str = None,
    limit: int = 10,
    memory = Depends(get_memory_manager)
):
    """
    Retrieve memories for a user.
    
    If query is provided, returns semantically similar memories.
    Otherwise, returns recent memories.
    """
    try:
        if query:
            memories = memory.retrieve_memories(
                user_id=user_id,
                query=query,
                n_results=limit
            )
        else:
            # Get all memories (limited)
            memories = memory.retrieve_memories(
                user_id=user_id,
                query="recent activities and goals",
                n_results=limit
            )
        
        return {
            "user_id": user_id,
            "memories": memories,
            "count": len(memories)
        }
        
    except Exception as e:
        logger.error(f"Get memories error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memory/{user_id}/stats", tags=["Memory"])
async def get_user_stats(
    user_id: str,
    memory = Depends(get_memory_manager)
):
    """Get statistics about a user's stored memories."""
    try:
        stats = memory.get_user_stats(user_id)
        return stats
    except Exception as e:
        logger.error(f"Get stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/memory/{user_id}", tags=["Memory"])
async def delete_user_memories(
    user_id: str,
    memory = Depends(get_memory_manager)
):
    """
    Delete all memories for a user (GDPR compliance).
    """
    try:
        count = memory.delete_user_memories(user_id)
        return {
            "status": "success",
            "deleted_count": count,
            "message": f"Deleted {count} memories for user {user_id}"
        }
    except Exception as e:
        logger.error(f"Delete memories error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================
# UTILITY ENDPOINTS
# =============================================

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    
    # Don't load memory on health check to keep it fast
    return HealthResponse(
        status="healthy",
        llm_provider=settings.llm_provider,
        memory_status="lazy_load"
    )


@app.get("/api/v1/prompt", tags=["Utility"])
async def get_system_prompt():
    """Get the current Nudge system prompt."""
    return {
        "system_prompt": NUDGE_SYSTEM_PROMPT,
        "description": "This is the personality prompt used for all Nudge responses"
    }


# Export app for Vercel
# Vercel expects the app to be named 'app'
__all__ = ["app"]

