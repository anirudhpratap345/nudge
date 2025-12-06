"""
Nudge Coach API - FastAPI Application

The single change that 10x's your AI coach:
- Long-term memory via ChromaDB + NV-Embed
- Nudge personality via system prompt (or fine-tuned LoRA)
- Fast inference via Groq (or self-hosted)
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import get_settings, NUDGE_SYSTEM_PROMPT
from models import (
    ChatRequest, ChatResponse,
    StoreMemoryRequest, MemoryEntry,
    UserProfile, HealthResponse
)

# Lazy imports to avoid slow startup
def get_memory_manager():
    from memory import get_memory_manager as _get_memory_manager
    return _get_memory_manager()

def get_llm():
    from llm import get_llm as _get_llm
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
    - **Long-term Memory**: Remembers everything about the user via ChromaDB + NV-Embed
    - **Personality**: Brutally specific advice, energy-matching, accountability questions
    - **Hinglish Support**: Automatically switches to Hinglish when user uses Hindi
    
    ## The Magic
    Before every response, we:
    1. Retrieve relevant memories from ChromaDB
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
        response_text = llm.generate(
            user_message=request.message,
            memory_context=memory_context,
            conversation_history=history_text
        )
        
        # ============================================
        # STEP 4: Store as New Memories
        # ============================================
        # Store user message
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
        
        return ChatResponse(
            response=response_text,
            user_id=request.user_id,
            memories_used=len(memories)
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Nudge Coach API",
        "version": "1.0.0",
        "description": "A sharp, caring, no-nonsense achievement coach",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# =============================================
# RUN WITH UVICORN
# =============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload for stability
    )

