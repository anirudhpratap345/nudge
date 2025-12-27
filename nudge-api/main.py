"""
Nudge Coach API - FastAPI Application

The single change that 10x's your AI coach:
- Long-term memory via ChromaDB + NV-Embed
- Nudge personality via system prompt (or fine-tuned LoRA)
- Fast inference via Groq (or self-hosted)
- Hybrid rule engine for 95%+ human-like responses
"""
import os
import sys

# Debug logging for container startup
print("=" * 60)
print("Starting FastAPI app...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print(f"Files in current dir: {os.listdir('.') if os.path.exists('.') else 'N/A'}")
print("=" * 60)

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from contextlib import asynccontextmanager
import logging
import random
import json
import os

from config import get_settings, NUDGE_SYSTEM_PROMPT, MCKENNA_SYSTEM_PROMPT
from models import (
    ChatRequest, ChatResponse,
    StoreMemoryRequest, MemoryEntry,
    UserProfile, HealthResponse,
    ImprovedNudgeRequest, ImprovedNudgeResponse, Message
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
    import re
    
    original_reply = reply
    
    # Validation: Check if response is empty or too short
    if not reply or not reply.strip() or len(reply.strip()) < 10:
        logger.warning("Empty or too short response, using fallback")
        actions = get_action_pool(memory_context)
        last_action = get_last_action(user_id, memory)
        available_actions = [a for a in actions if a.lower() != last_action.lower()]
        if not available_actions:
            available_actions = actions
        new_action = random.choice(available_actions)
        identity = extract_identity(memory_context)
        reply = f"As you're becoming {identity}, {new_action}. Done? Yes/No"
        store_last_action(user_id, new_action, memory)
        return reply
    
    # Normalize whitespace
    reply = " ".join(reply.split())
    
    # Rule 1: Ban generics — use whole word matching to avoid false positives
    bad_patterns = [
        r'\baffirmation\b', r'\bbrainstorm\b', r'\breflect\b', r'\bjournal\b', 
        r'\bsticky note\b', r'\bfailure log\b', r'\bmission statement\b', 
        r'\bplan a block\b', r'\bwrite down\b', r'\bthink about\b', 
        r'\bvisualize\b', r'\bimagine\b', r'\bmeditate\b', r'\bbreathe\b', 
        r'\bgratitude\b', r'\bconsider\b'
    ]
    
    # Check if reply contains bad patterns (whole words only)
    has_bad_pattern = False
    for pattern in bad_patterns:
        if re.search(pattern, reply.lower()):
            has_bad_pattern = True
            break
    
    if has_bad_pattern:
        # Extract any good action from the reply before replacing
        # Look for action-like phrases (imperative verbs + objects)
        action_pattern = r'(?:open|solve|write|create|add|push|run|implement|build|fix|refactor|update|test|deploy)\s+[^.!?]+'
        existing_actions = re.findall(action_pattern, reply, re.IGNORECASE)
        
        if existing_actions and len(existing_actions[0]) > 15:
            # Preserve the action, just clean up the generic parts
            good_action = existing_actions[0].strip()
            identity = extract_identity(memory_context)
            reply = f"As you're becoming {identity}, {good_action}. Done? Yes/No"
            store_last_action(user_id, good_action, memory)
            logger.info(f"Rule engine: Preserved action, removed generics: {good_action[:50]}")
        else:
            # No good action found, replace with fresh one
            actions = get_action_pool(memory_context)
            last_action = get_last_action(user_id, memory)
            available_actions = [a for a in actions if a.lower() != last_action.lower()]
            if not available_actions:
                available_actions = actions
            new_action = random.choice(available_actions)
            identity = extract_identity(memory_context)
            reply = f"As you're becoming {identity}, {new_action}. Done? Yes/No"
            store_last_action(user_id, new_action, memory)
            logger.info(f"Rule engine: Replaced generic response with action: {new_action[:50]}")
    
    # Rule 2: Inject wit if low energy (but preserve existing good content)
    low_energy_words = ["low", "stuck", "fail", "negative", "tired", "exhausted", "burnt", "depressed", "sad"]
    if any(word in user_message.lower() for word in low_energy_words):
        if "As you're becoming" not in reply:
            # Extract action from reply if present
            action_pattern = r'(?:open|solve|write|create|add|push|run|implement|build|fix|refactor|update|test|deploy)\s+[^.!?]+'
            existing_actions = re.findall(action_pattern, reply, re.IGNORECASE)
            
            if existing_actions:
                good_action = existing_actions[0].strip()
                identity = extract_identity(memory_context, low_energy=True)
                reply = f"As you're becoming {identity}, {good_action}. Done? Yes/No"
            else:
                # No action found, add one
                actions = get_action_pool(memory_context)
                last_action = get_last_action(user_id, memory)
                available_actions = [a for a in actions if a.lower() != last_action.lower()]
                if not available_actions:
                    available_actions = actions
                new_action = random.choice(available_actions)
                identity = extract_identity(memory_context, low_energy=True)
                reply = f"As you're becoming {identity}, {new_action}. Done? Yes/No"
                store_last_action(user_id, new_action, memory)
    
    # Rule 3: Ensure Yes/No end (but check for duplication first)
    reply_stripped = reply.rstrip()
    has_yes_no = reply_stripped.endswith("Yes/No") or reply_stripped.endswith("Yes/No.")
    has_done = "Done?" in reply_stripped[-20:]  # Check last 20 chars
    
    if not has_yes_no:
        if has_done:
            # Already has "Done?" but missing "Yes/No"
            reply = reply_stripped.rstrip(".") + " Yes/No"
        else:
            # Add both
            reply = reply_stripped.rstrip(".") + ". Done? Yes/No"
    
    # Rule 4: Remove internal questions (but preserve rhetorical ones and emphasis)
    # Only remove actual queries, not rhetorical questions
    lines = reply.split("\n")
    cleaned_lines = []
    for line in lines:
        # Check if it's a real question (starts with question words)
        is_real_question = re.match(r'^\s*(what|how|why|when|where|who|which|can|could|should|would|will|do|does|did|is|are|was|were)\s+', line.lower())
        has_question_mark = "?" in line
        is_final_yes_no = line.strip().endswith("Yes/No") or line.strip().endswith("Yes/No.")
        
        if has_question_mark and is_real_question and not is_final_yes_no:
            # Remove question mark from real queries
            line = line.replace("?", "").strip()
        cleaned_lines.append(line)
    reply = "\n".join(cleaned_lines)
    
    # Rule 5: Ensure action is fresh (not repeated) - use better matching
    last_action = get_last_action(user_id, memory)
    if last_action and len(last_action) > 10:
        # Use word overlap to detect similarity, not exact substring
        last_words = set(last_action.lower().split())
        reply_words = set(reply.lower().split())
        overlap = len(last_words & reply_words)
        similarity = overlap / max(len(last_words), 1)
        
        # If >60% word overlap and reply is short, likely a repeat
        if similarity > 0.6 and len(reply.split()) < 25:
            actions = get_action_pool(memory_context)
            available_actions = [a for a in actions if a.lower() != last_action.lower()]
            if available_actions:
                new_action = random.choice(available_actions)
                store_last_action(user_id, new_action, memory)
                # Preserve identity if present
                if "As you're becoming" in reply:
                    identity_part = reply.split(",")[0] if "," in reply else "As you're becoming the badass who ships daily"
                    reply = f"{identity_part}, {new_action}. Done? Yes/No"
                else:
                    reply = f"{reply.split('.')[0]}. {new_action}. Done? Yes/No"
                logger.info(f"Rule engine: Replaced repeated action with: {new_action[:50]}")
    
    # Rule 6: Extract and store action for next time (improved extraction)
    if reply:
        # Try multiple patterns to extract action
        action_patterns = [
            r'As you\'re becoming[^,]+,([^.!?]+)',  # After "As you're becoming..., action"
            r'(?:open|solve|write|create|add|push|run|implement|build|fix|refactor|update|test|deploy)\s+[^.!?]+',  # Imperative action
            r'\.\s*([A-Z][^.!?]+?)(?:\.|Done)',  # Sentence before "Done"
        ]
        
        extracted_action = None
        for pattern in action_patterns:
            matches = re.findall(pattern, reply, re.IGNORECASE)
            if matches:
                extracted_action = matches[0].strip()
                if len(extracted_action) > 10 and len(extracted_action) < 200:
                    break
        
        if extracted_action:
            store_last_action(user_id, extracted_action, memory)
    
    if original_reply != reply:
        logger.info(f"Rule engine applied: {len(original_reply)} -> {len(reply)} chars")
    
    return reply


def extract_identity(memory_context: str, low_energy: bool = False) -> str:
    """Extract user identity from memory context with better matching"""
    import re
    
    context_lower = memory_context.lower()
    
    # More sophisticated identity extraction
    if low_energy:
        return "the badass who ships despite everything"
    
    # Check for specific identities (whole phrases, not just keywords)
    if re.search(r'\b(founder|startup|entrepreneur|building|launching)\b', context_lower):
        return "the founder who ships daily"
    elif re.search(r'\b(ai|ml|machine learning|artificial intelligence|data scientist)\b', context_lower):
        return "the AI engineer who ships daily"
    elif re.search(r'\b(engineer|developer|programmer|coder|software|backend|frontend|full.?stack)\b', context_lower):
        return "the engineer who ships daily"
    elif re.search(r'\b(product manager|pm|product)\b', context_lower):
        return "the PM who ships daily"
    else:
        return "the badass who ships daily"


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

# Mount static files directory (for serving frontend)
# Try multiple paths: container (./static), local dev (nudge-agent/static), or parent (../nudge-agent/static)
static_paths = [
    os.path.join(os.path.dirname(__file__), "static"),  # Container: ./static
    os.path.join(os.path.dirname(__file__), "nudge-agent", "static"),  # Local: nudge-agent/static
    os.path.join(os.path.dirname(__file__), "..", "nudge-agent", "static"),  # Alternative
]

static_dir = None
for path in static_paths:
    if os.path.exists(path):
        static_dir = path
        break

if static_dir:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    print(f"✓ Static files mounted from: {static_dir}")
else:
    print("⚠ Warning: Static directory not found, static files will not be served")


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
            
            # Validate response before rule engine
            if not response_text or not response_text.strip():
                logger.warning("LLM returned empty response, using fallback")
                response_text = "Try again—connection hiccup. Done? Yes/No"
            elif len(response_text.strip()) < 10:
                logger.warning("LLM returned too short response, using fallback")
                response_text = "Try again—connection hiccup. Done? Yes/No"
                
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
# IMPROVED ADVISOR NUDGE ENDPOINT
# =============================================

@app.post("/api/v1/improved-advisor-nudge", response_model=ImprovedNudgeResponse, tags=["Coaching"])
async def improved_advisor_nudge(
    request: ImprovedNudgeRequest,
    memory = Depends(get_memory_manager),
    llm = Depends(get_llm)
):
    """
    Generate personalized nudge with McKenna-style hypnotic language.
    
    This endpoint delivers Unlimits' core value:
    - Micro-action for TODAY (≤10 min)
    - Hypnotic visualization (sensory-rich future self)
    - Deep personalization (progress + personality adaptive)
    """
    try:
        # 1. Retrieve deep context from memory
        memories = memory.retrieve_memories(
            user_id=request.user_id,
            query=request.dream,
            n_results=10
        )
        memory_context = memory.format_memory_context(memories)
        
        # 2. FORCED INDECISION DETECTION (Check BEFORE anything else!)
        dream_lower = request.dream.lower()
        
        # Strong indecision signals - check explicitly
        indecision_signals = [
            " or ",           # "Meta or quant"
            " vs ",           # "FAANG vs startup"  
            "should i",       # "Should I become..."
            "also ",          # "also thinking"
            "considering",    # "considering both"
            "which ",         # "which path"
            "thinking about", # "thinking about grad school"
            "between ",       # "between these options"
        ]
        
        has_indecision = any(signal in dream_lower for signal in indecision_signals)
        
        if has_indecision:
            logger.warning(f"⚠️  INDECISION DETECTED for user {request.user_id}: '{request.dream[:60]}...'")
        
        # 3. Detect full context state (indecision, shift, or single goal)
        context_state = _detect_context_state(request.dream, memories)
        # Override with our explicit detection
        if has_indecision:
            context_state["has_indecision"] = True
            context_state["state"] = "indecision"
        
        logger.info(f"User {request.user_id} context state: {context_state['state']}, indecision={context_state['has_indecision']}")
        
        # 4. Generate appropriate visualization title
        visualization_title = _generate_visualization_title(request.dream, context_state)
        logger.info(f"User {request.user_id} visualization title: {visualization_title}")
        
        # 5. Build progress summary
        progress_summary = _build_progress_summary(request.progress)
        
        # 6. Extract/format personality
        personality_traits = _extract_personality(
            request.personality,
            memories
        )
        
        # 6.5. Get deep personality profile from memory
        try:
            personality_profile = memory.get_personality_profile(request.user_id)
            if personality_profile and personality_profile.get("status") != "new_user":
                if personality_profile.get("energy_patterns"):
                    personality_traits["energy_pattern"] = personality_profile["energy_patterns"]
                if personality_profile.get("communication_preference"):
                    personality_traits["style"] = personality_profile["communication_preference"]
                if personality_profile.get("struggle_themes"):
                    personality_traits["struggles"] = personality_profile["struggle_themes"]
        except Exception as e:
            logger.warning(f"Failed to get personality profile: {e}")
        
        # Format personality traits as string for prompt
        personality_str = ", ".join([f"{k}: {v}" for k, v in personality_traits.items()]) if personality_traits else "exploring preferences"
        
        # 7. Build McKenna prompt with personality AND context adaptation
        from datetime import datetime
        try:
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            today_date = datetime.now(ist).strftime("%A, %B %d, %Y")
        except ImportError:
            today_date = datetime.now().strftime("%A, %B %d, %Y")
        
        # Extract energy_level and preferred_style
        energy_level = request.personality.get("energy_level", "moderate") if request.personality else "moderate"
        preferred_style = request.personality.get("preferred_style", "balanced") if request.personality else "balanced"
        
        logger.info(f"User {request.user_id}: dream='{request.dream[:50]}...', energy={energy_level}, style={preferred_style}")
        
        # 8. BUILD CONTEXT STATE DESCRIPTION WITH CRITICAL OVERRIDE FOR INDECISION
        if has_indecision or context_state.get("has_indecision"):
            # Extract the options being compared
            options = context_state.get("options", [])
            options_str = " vs ".join(options) if options else "multiple options"
            
            context_state_description = f"""

🚨🚨🚨 CRITICAL OVERRIDE - INDECISION DETECTED 🚨🚨🚨

THE USER IS EXPRESSING INDECISION / EXPLORING MULTIPLE OPTIONS.
They are considering: {options_str}

YOU MUST:
1. ✅ Start visualization title with "Exploring:" (e.g., "Exploring: {options_str}")
2. ✅ Address ALL options they mentioned (don't pick one)
3. ✅ Provide a comparison/decision framework in nudge (pros/cons table, decision matrix)
4. ✅ Acknowledge uncertainty in visualization ("standing at a crossroads...")

YOU MUST NOT:
❌ Treat this as a single goal
❌ Push toward one option and ignore others
❌ Give a generic nudge like "make a commit" 
❌ Use title starting with "Journey to:"

Current user state: EXPLORING / UNDECIDED
Options being considered: {options_str}

CORRECT NUDGE EXAMPLE: "Create a 2-column comparison table for your options. List 3 pros for each. Set a 10-minute timer."
"""
        else:
            context_state_description = context_state.get("description", "Focus on this single, clear goal.")
        
        mckenna_prompt = MCKENNA_SYSTEM_PROMPT.format(
            dream=request.dream,
            progress_summary=progress_summary,
            personality_traits=personality_str,
            memory_context=memory_context if memory_context else "(No memories yet - this is a new user)",
            today_date=today_date,
            energy_level=energy_level,
            preferred_style=preferred_style,
            context_state=context_state_description,
            visualization_title=visualization_title
        )
        
        # 5. Format conversation history if provided
        conversation_history = ""
        if request.history:
            conversation_history = "\n".join([
                f"{msg.role.value.capitalize()}: {msg.content}"
                for msg in request.history[-5:]  # Last 5 messages
            ])
        
        # 6. Generate response with McKenna voice
        user_query = f"Give me today's nudge to move toward: {request.dream}"
        
        # Use existing LLM.generate() method but with custom system prompt
        # We'll need to temporarily override the system prompt
        original_prompt = NUDGE_SYSTEM_PROMPT
        
        # Create a custom prompt builder for this call
        system_prompt = mckenna_prompt
        full_context = f"{conversation_history}\n\nUser: {user_query}" if conversation_history else f"User: {user_query}"
        
        # Build messages for Groq API directly (since we need custom system prompt)
        if hasattr(llm, 'client'):
            # Groq LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_context}
            ]
            
            try:
                response = llm.client.chat.completions.create(
                    model=llm.settings.groq_model,
                    messages=messages,
                    temperature=0.7,  # Higher for more varied, creative responses
                    max_tokens=600,   # More tokens for full visualization
                    top_p=0.9
                )
                response_text = response.choices[0].message.content
                logger.info(f"LLM response for {request.user_id}: {response_text[:100]}...")
            except Exception as e:
                logger.error(f"Groq API error in improved_advisor_nudge: {e}")
                # Dynamic fallback based on dream, personality, AND context state
                fallback_nudge = _generate_fallback_nudge(request.dream, energy_level, context_state)
                fallback_viz = _generate_fallback_visualization(request.dream, energy_level)
                response_text = f"NUDGE: {fallback_nudge}\n\nVISUALIZATION: {fallback_viz}"
        else:
            # Fallback: use regular generate method
            response_text = llm.generate(
                user_message=user_query,
                memory_context=memory_context,
                conversation_history=conversation_history
            )
        
        # 7. Parse into nudge + visualization
        parsed = _parse_mckenna_response(response_text)
        
        # 8. Generate structured visualization (enhanced version) - pass energy/style and context-aware title
        try:
            # Ensure energy and style are in personality_traits for visualization
            viz_personality = personality_traits.copy()
            viz_personality["energy"] = energy_level
            viz_personality["style"] = preferred_style
            viz_personality["context_state"] = context_state.get("state", "single_goal")
            
            structured_viz = generate_structured_visualization(
                dream=request.dream,
                personality=viz_personality,
                llm=llm
            )
            
            # Use the context-aware visualization title instead of default
            parsed["visualization"] = {
                "title": visualization_title,  # Use context-aware title!
                "phases": structured_viz["phases"],
                "steps": structured_viz["steps"],
                "full_text": structured_viz["full_text"],
                "duration_seconds": structured_viz["total_duration"]
            }
        except Exception as e:
            logger.warning(f"Failed to generate structured visualization, using parsed version: {e}")
            # Continue with parsed visualization if structured generation fails
        
        # 9. Store interaction in memory (store nudge only, not dream to avoid pollution)
        try:
            memory.store_memory(
                user_id=request.user_id,
                content=f"Coaching session: Received nudge '{parsed['nudge'][:100]}...'",
                memory_type="coaching"
            )
        except Exception as e:
            logger.warning(f"Failed to store memory: {e}")
        
        return ImprovedNudgeResponse(
            nudge=parsed["nudge"],
            visualization=parsed["visualization"],
            personality_insights=personality_traits,
            tts_ready=False  # Set to True when TTS implemented
        )
        
    except Exception as e:
        logger.error(f"Error in improved_advisor_nudge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for improved advisor nudge
def _build_progress_summary(progress: dict) -> str:
    """Format progress metrics into narrative"""
    if not progress:
        return "Just starting the journey"
    
    days = progress.get("days_active", 0)
    wins = progress.get("wins", 0)
    struggles = progress.get("struggles", [])
    
    summary = f"{days} days active, {wins} wins achieved"
    if struggles:
        summary += f". Current challenges: {', '.join(struggles[:3])}"  # Limit to 3
    
    return summary


def _extract_personality(
    personality: dict,
    memories: list
) -> dict:
    """Extract personality traits from input + memory patterns"""
    traits = {}
    
    # From explicit input
    if personality.get("energy_level"):
        traits["energy"] = personality["energy_level"]
    if personality.get("preferred_style"):
        traits["style"] = personality["preferred_style"]
    
    # From memory patterns (simple keyword extraction)
    if memories:
        memory_text = " ".join([
            str(m.get("content", m.get("text", ""))) 
            for m in memories[:5]  # Limit to first 5
        ]).lower()
        
        if "overwhelm" in memory_text or "stuck" in memory_text:
            traits["tends_toward"] = "overwhelm"
        elif "excit" in memory_text or "motivat" in memory_text:
            traits["tends_toward"] = "high_enthusiasm"
        elif "tired" in memory_text or "burnt" in memory_text:
            traits["tends_toward"] = "low_energy"
    
    return traits if traits else {"status": "exploring_preferences"}


def generate_structured_visualization(
    dream: str,
    personality: dict,
    llm
) -> dict:
    """
    Generate a structured McKenna-style visualization with specific phases.
    ADAPTS to energy_level and preferred_style for unique, personalized output.
    """
    # Extract personality for tone adaptation
    energy_level = personality.get("energy", personality.get("energy_level", "moderate"))
    preferred_style = personality.get("style", personality.get("preferred_style", "balanced"))
    personality_str = ", ".join([f"{k}: {v}" for k, v in personality.items()]) if personality else "exploring preferences"
    
    # Build adaptive instructions based on energy level
    if energy_level == "low":
        tone_instruction = """
TONE: Gentle, calming, nurturing. Use soft language:
- "Gently allow yourself to...", "Softly notice...", "Let yourself drift into..."
- Slower pace, more grounding, permission-giving language
- Avoid: "surge", "power", "unstoppable", aggressive language"""
    elif energy_level == "high":
        tone_instruction = """
TONE: Energetic, dynamic, powerful. Use vibrant language:
- "Feel the surge of...", "You're unstoppable...", "Tap into your fire..."
- Fast-paced, action-oriented, empowering
- Use: "Launch", "Conquer", "Dominate", "Crush it" """
    else:
        tone_instruction = """
TONE: Balanced, confident, warm. Use clear language:
- "Notice...", "Feel...", "See yourself..."
- Steady pace, mix of grounding and action"""
    
    # Build style instructions
    if preferred_style == "gentle":
        style_instruction = "Be warm and nurturing. Use invitations, not commands. More imagery, less instruction."
    elif preferred_style == "direct":
        style_instruction = "Be concise and action-focused. Clear commands. Skip excessive imagery."
    else:
        style_instruction = "Balance warmth with clarity. Brief grounding, then clear visualization."
    
    viz_prompt = f"""Create a 90-second guided visualization SPECIFICALLY for this dream: {dream}

CRITICAL: This visualization must be UNIQUE to the dream "{dream}" - reference specific elements of this goal.

{tone_instruction}

Style: {style_instruction}

Structure it in 4 phases:
1. GROUND (15s): Breath awareness, present moment anchoring
2. IMAGINE (30s): Vivid sensory details of achieving "{dream}" specifically - what do they see, hear, feel?
3. EMBODY (30s): Identity shift - "I AM the person who has {dream}" - use present tense
4. COMMIT (15s): One immediate micro-action toward "{dream}"

Format each phase on a new line: "PHASE_NAME: [content]"
Make it UNIQUE to this specific dream and energy level."""

    # Use LLM to generate visualization
    system_prompt = f"""You are Paul McKenna's AI assistant. Create a sensory-rich visualization.
CRITICAL: The dream is "{dream}" - make the visualization SPECIFIC to this exact goal.
Energy level is {energy_level}. Adapt your language accordingly."""
    
    # Build messages for Groq API
    if hasattr(llm, 'client'):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": viz_prompt}
        ]
        
        try:
            response = llm.client.chat.completions.create(
                model=llm.settings.groq_model,
                messages=messages,
                temperature=0.8,  # Higher for unique, creative visualizations
                max_tokens=400,
                top_p=0.9
            )
            viz_response = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            # Dynamic fallback based on dream and energy
            viz_response = _generate_fallback_visualization(dream, energy_level)
    else:
        viz_response = _generate_fallback_visualization(dream, energy_level)
    
    # Parse phases
    phases = {}
    phase_order = ["GROUND", "IMAGINE", "EMBODY", "COMMIT"]
    
    for line in viz_response.split("\n"):
        line = line.strip()
        if not line:
            continue
        for phase in phase_order:
            if line.upper().startswith(phase):
                phase_text = line.split(":", 1)[1].strip() if ":" in line else line.replace(phase, "").strip()
                phases[phase] = phase_text
                break
    
    # Fill missing phases with DYNAMIC fallbacks that use the dream
    for phase in phase_order:
        if phase not in phases:
            phases[phase] = _get_dynamic_phase_fallback(phase, dream, energy_level)
    
    steps = [
        {"text": phases["GROUND"], "duration_seconds": 15},
        {"text": phases["IMAGINE"], "duration_seconds": 30},
        {"text": phases["EMBODY"], "duration_seconds": 30},
        {"text": phases["COMMIT"], "duration_seconds": 15}
    ]
    
    return {
        "title": f"Journey to: {dream[:50]}",
        "phases": phases,
        "steps": steps,
        "total_duration": 90,
        "full_text": viz_response
    }


def _detect_context_state(dream: str, memories: list = None) -> dict:
    """
    Detect the user's context state: indecision, context shift, or single goal.
    Returns context info for prompt and visualization title.
    """
    dream_lower = dream.lower()
    
    # Indecision signals
    indecision_signals = ["or", " vs ", "also", "thinking about", "should i", "considering", "between", "either"]
    has_indecision = any(signal in dream_lower for signal in indecision_signals)
    
    # Extract options if indecision detected
    options = []
    if has_indecision:
        # Try to extract the options being compared
        for splitter in [" or ", " vs ", " between "]:
            if splitter in dream_lower:
                parts = dream_lower.split(splitter)
                options = [p.strip()[:40] for p in parts if len(p.strip()) > 3]
                break
        
        # If "also" or "thinking about", extract what they're considering
        if not options and ("also" in dream_lower or "thinking" in dream_lower):
            options = ["current path", "new option being considered"]
    
    # Context shift detection (if memories exist)
    context_shift = False
    prev_focus = None
    if memories and len(memories) > 0:
        prev_content = " ".join([str(m.get("content", m.get("text", "")))[:100] for m in memories[:3]]).lower()
        
        # Extract key words from current dream and memories
        dream_words = set(w for w in dream_lower.split() if len(w) > 3)
        memory_words = set(w for w in prev_content.split() if len(w) > 3)
        
        # If current dream has many new words not in memories, it's likely a shift
        new_words = dream_words - memory_words
        if len(new_words) > 5:
            context_shift = True
            # Try to identify what the previous focus was
            if "faang" in prev_content or "leetcode" in prev_content or "interview" in prev_content:
                prev_focus = "FAANG prep"
            elif "startup" in prev_content or "launch" in prev_content or "saas" in prev_content:
                prev_focus = "startup launch"
            elif "engineer" in prev_content or "developer" in prev_content:
                prev_focus = "engineering career"
            else:
                prev_focus = "previous goal"
    
    # Determine state
    if has_indecision:
        state = "indecision"
        state_description = f"""
CRITICAL CONTEXT: User is in EXPLORATION/INDECISION mode.
They are considering multiple options: {', '.join(options) if options else 'multiple paths'}

DO NOT:
- Just pick one option and ignore the others
- Repeat a single goal as if they've decided
- Give a generic "make a commit" type nudge

DO:
- Help them COMPARE their options
- Provide a decision-making framework
- Suggest creating a pros/cons table or talking to people in each field
- Acknowledge that exploration is valid
"""
    elif context_shift and prev_focus:
        state = "context_shift"
        state_description = f"""
CRITICAL CONTEXT: User's focus has SHIFTED from their previous context.
Previous focus was: {prev_focus}
Current message indicates: {dream[:60]}

Address their CURRENT state. The previous context is just background.
If appropriate, acknowledge the transition.
"""
    else:
        state = "single_goal"
        state_description = "Focus on this single, clear goal."
    
    return {
        "state": state,
        "description": state_description,
        "has_indecision": has_indecision,
        "options": options,
        "context_shift": context_shift,
        "prev_focus": prev_focus
    }


def _generate_visualization_title(dream: str, context_state: dict) -> str:
    """Generate an appropriate visualization title based on context state."""
    state = context_state.get("state", "single_goal")
    
    if state == "indecision":
        options = context_state.get("options", [])
        if len(options) >= 2:
            return f"Exploring: {options[0].title()} vs {options[1].title()}"
        else:
            return f"Exploring: Your Options and Possibilities"
    
    elif state == "context_shift":
        prev_focus = context_state.get("prev_focus", "previous path")
        # Extract current focus from dream
        current_focus = dream[:35].strip()
        return f"Transitioning: {prev_focus} → {current_focus}"
    
    else:
        # Single goal - clean it up for title
        clean_dream = dream[:50].strip()
        return f"Journey to: {clean_dream}"


def _generate_fallback_nudge(dream: str, energy_level: str, context_state: dict = None) -> str:
    """
    Generate context-aware fallback nudges based on:
    1. Context state (indecision, shift, single goal)
    2. Keywords in dream (career, FAANG, transition, launch, etc.)
    3. Indecision signals (or, also, thinking, should I)
    4. Energy level (low/moderate/high)
    """
    dream_lower = dream.lower()
    
    # CATEGORY 0: Indecision / Exploration (highest priority)
    if context_state and context_state.get("has_indecision"):
        options = context_state.get("options", [])
        if options:
            action = f"Create a 2-column comparison: '{options[0][:20]}' vs '{options[1][:20] if len(options) > 1 else 'option 2'}'. List 3 pros for each. Takes 10 minutes."
        else:
            action = "Open a notes app and create a decision matrix: List your options as columns, then add rows for 'daily work', 'growth potential', 'lifestyle impact'. Score each 1-5."
        
        if energy_level == "low":
            return f"Gently explore your options: {action}"
        elif energy_level == "high":
            return f"Let's get clarity! {action}"
        else:
            return f"Time for clarity: {action}"
    
    # CATEGORY 1: FAANG / Tech Interviews
    faang_keywords = ["meta", "google", "faang", "leetcode", "interview", "amazon", "apple", "microsoft", "netflix"]
    if any(kw in dream_lower for kw in faang_keywords):
        actions = [
            "Open LeetCode and solve problem #1 (Two Sum) with optimal solution. Time yourself: 15 minutes max.",
            "Open a Google Doc and write 3 bullet points explaining how you'd design a URL shortener (system design prep).",
            "Search 'Meta software engineer interview questions' and read through the top 3 Glassdoor posts. Note 2 common themes."
        ]
        action = actions[hash(dream) % len(actions)]
    
    # CATEGORY 2: Startup / Launch / Product
    elif any(kw in dream_lower for kw in ["startup", "launch", "saas", "product", "customer", "mvp", "ship"]):
        actions = [
            "Open your project and ship ONE small improvement - even a typo fix counts. Push it live.",
            "Create a file called 'launch-checklist.md' and write 5 things needed before first customer. Pick the smallest one.",
            "Open Twitter/X and write a tweet describing your product in one sentence. Don't post yet - just draft it."
        ]
        action = actions[hash(dream) % len(actions)]
    
    # CATEGORY 3: Career Transition
    elif any(kw in dream_lower for kw in ["become", "transition", "switch to", "move to", "career change", "from", "to"]):
        actions = [
            "Open LinkedIn and find 3 people in your target role. Note what skills they highlight in their profiles.",
            "Search YouTube for 'day in the life of [your target role]'. Watch one video (10 min) and note 2 surprises.",
            "Create a 'skills gap' list: Write 3 skills you have, 3 you need for the target role. Circle the easiest to learn."
        ]
        action = actions[hash(dream) % len(actions)]
    
    # CATEGORY 4: Burnout / Recovery / Low Energy
    elif any(kw in dream_lower for kw in ["burned out", "burnt out", "tired", "exhausted", "overwhelmed", "stuck", "lost", "energy"]):
        actions = [
            "Step away from screen for 5 minutes. Walk to window, take 10 slow breaths. Then write ONE tiny task you can do today.",
            "Open your task list and DELETE 3 things that don't actually matter this week. Feel the relief.",
            "Set a timer for 10 minutes. Do ONE small task - anything. When timer rings, you're done. No guilt."
        ]
        action = actions[hash(dream) % len(actions)]
    
    # CATEGORY 5: Skill Building / Learning
    elif any(kw in dream_lower for kw in ["learn", "master", "improve", "practice", "study", "course", "tutorial"]):
        actions = [
            "Open YouTube and search for a 10-minute tutorial on your target skill. Watch it actively - pause and try things.",
            "Create a 'learning log' file. Write today's date and ONE concept you want to understand better. Add notes as you learn.",
            "Find one code example or tutorial and type it out yourself (don't copy-paste). Understanding comes through fingers."
        ]
        action = actions[hash(dream) % len(actions)]
    
    # CATEGORY 6: General Engineering / Coding
    elif any(kw in dream_lower for kw in ["engineer", "developer", "coding", "programming", "software", "code"]):
        actions = [
            "Open your IDE and write or review 10 lines of code. Doesn't matter what - just touch the code.",
            "Pick one function in your codebase. Add one comment explaining what it does. Or improve one variable name.",
            "Open GitHub and star one repo related to your goals. Skim the README. Note one thing you could learn from it."
        ]
        action = actions[hash(dream) % len(actions)]
    
    # CATEGORY 7: Quant / Finance / Trading
    elif any(kw in dream_lower for kw in ["quant", "trading", "finance", "hedge fund", "algo", "quantitative"]):
        actions = [
            "Open a Python notebook and import pandas. Load any stock data (yfinance) and calculate a simple moving average.",
            "Search 'quant developer interview questions' and read through 3. Note which math concepts keep appearing.",
            "Find one quantitative finance blog (QuantStart, Quantopian) and read one recent article. Note one new concept."
        ]
        action = actions[hash(dream) % len(actions)]
    
    # DEFAULT: Generic but still actionable
    else:
        action = f"Open a notes app and write: 'My next concrete step toward {dream[:25]}... is [blank]'. Fill in the blank with something you can do in 10 minutes."
    
    # Energy level adaptation
    if energy_level == "low":
        prefix = "Gently, when you're ready: "
        suffix = " No pressure - even starting counts."
    elif energy_level == "high":
        prefix = "Let's GO! "
        suffix = " You've got this!"
    else:
        prefix = "Here's your 10-minute action: "
        suffix = ""
    
    return f"{prefix}{action}{suffix}"


def _generate_fallback_visualization(dream: str, energy_level: str) -> str:
    """Generate a dynamic fallback visualization based on dream and energy."""
    if energy_level == "low":
        ground = "Close your eyes. Allow three gentle breaths. Feel the weight of your body settling. You are safe here."
        imagine = f"Softly, see yourself having achieved '{dream[:40]}'. Notice how calm and peaceful you feel. The struggle is behind you."
        embody = f"Whisper to yourself: 'I am becoming this person.' Feel the gentle truth of it. This is who you are growing into."
        commit = "When ready, open your eyes. Take one tiny step. Even small movements matter."
    elif energy_level == "high":
        ground = "Take a powerful breath. Feel your energy rising. You're READY for this."
        imagine = f"SEE yourself crushing '{dream[:40]}'! Feel that surge of victory! You're unstoppable!"
        embody = f"You ARE the person who achieves this! Feel it in your bones! This is your DESTINY!"
        commit = "Open your eyes and MOVE! Take one action RIGHT NOW toward your dream!"
    else:
        ground = "Close your eyes. Take three deep breaths. Feel your body relaxing, your mind clearing."
        imagine = f"See yourself achieving '{dream[:40]}'. Notice the details - what you see, hear, feel. You're there."
        embody = f"You ARE becoming this person. Feel this identity settling into your bones. This is who you are."
        commit = "Open your eyes. Take one clear action today that moves you toward this reality."
    
    return f"""GROUND: {ground}
IMAGINE: {imagine}
EMBODY: {embody}
COMMIT: {commit}"""


def _get_dynamic_phase_fallback(phase: str, dream: str, energy_level: str) -> str:
    """Get a dynamic fallback for a specific phase."""
    if phase == "GROUND":
        if energy_level == "low":
            return "Gently close your eyes. Let three soft breaths release any tension. You are safe and supported."
        elif energy_level == "high":
            return "Take a powerful breath! Feel your energy rising! You're about to visualize your victory!"
        else:
            return "Close your eyes. Breathe deeply three times. Feel yourself becoming present, grounded, ready."
    
    elif phase == "IMAGINE":
        if energy_level == "low":
            return f"Softly picture yourself having achieved '{dream[:35]}'. Feel the peaceful satisfaction. See how far you've come."
        elif energy_level == "high":
            return f"VIVIDLY see yourself CRUSHING '{dream[:35]}'! Feel that rush of triumph! You made it happen!"
        else:
            return f"See yourself achieving '{dream[:35]}'. Notice every detail - what you see, hear, feel. You're living it."
    
    elif phase == "EMBODY":
        if energy_level == "low":
            return f"Gently repeat: 'I am becoming this person.' Let this truth settle softly into your heart."
        elif energy_level == "high":
            return f"Say it with POWER: 'I AM this person!' Feel unstoppable confidence surge through you!"
        else:
            return f"Feel yourself becoming this person. Your identity is shifting. This is who you ARE now."
    
    elif phase == "COMMIT":
        if energy_level == "low":
            return "When ready, gently open your eyes. Choose one small step you can take today. Even tiny progress counts."
        elif energy_level == "high":
            return "OPEN YOUR EYES! You're on FIRE! Take one bold action RIGHT NOW! GO!"
        else:
            return "Open your eyes with calm confidence. Take one clear action today toward your dream."
    
    return f"Focus on {dream[:30]} and take one step forward."


def _parse_mckenna_response(response: str) -> dict:
    """Parse LLM response into structured nudge + visualization"""
    import re
    
    # Clean up markdown formatting
    response = re.sub(r'\*{2,}', '', response)  # Remove ** and ****
    response = response.strip()
    
    # Try to split on explicit markers first
    parts = re.split(r'VISUALIZATION:', response, flags=re.IGNORECASE)
    
    if len(parts) == 2:
        nudge_raw = parts[0].strip()
        viz_text = parts[1].strip()
        
        # Clean nudge: remove "NUDGE:" header and any markdown
        nudge = re.sub(r'^NUDGE:?\s*', '', nudge_raw, flags=re.IGNORECASE).strip()
        nudge = re.sub(r'\*+', '', nudge).strip()  # Remove any remaining markdown
    else:
        # Fallback: try to find nudge in first paragraph
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        
        if len(paragraphs) >= 2:
            # First paragraph is likely nudge
            nudge = paragraphs[0].strip()
            viz_text = "\n\n".join(paragraphs[1:]).strip()
        elif len(paragraphs) == 1:
            # Single paragraph - try to split by sentence
            sentences = [s.strip() for s in response.split(". ") if s.strip()]
            if len(sentences) >= 3:
                nudge = ". ".join(sentences[:2]) + "."
                viz_text = ". ".join(sentences[2:])
            else:
                # Use entire response as nudge, generate fallback viz
                nudge = response
                viz_text = "Close your eyes. Feel your breath slowing. See yourself achieving your dream."
        else:
            nudge = response
            viz_text = "Close your eyes. Feel your breath slowing. See yourself achieving your dream."
    
    # Clean up nudge: remove headers, incomplete sentences, markdown
    nudge = re.sub(r'^(Today\'s|Your|Nudge|NUDGE)[:\s]*', '', nudge, flags=re.IGNORECASE).strip()
    nudge = re.sub(r'\*+', '', nudge).strip()
    
    # Remove incomplete sentences (ending with ":" or "?" without completion)
    nudge_lines = nudge.split('\n')
    cleaned_lines = []
    for line in nudge_lines:
        line = line.strip()
        # Skip lines that are just headers or incomplete
        if line and not line.endswith(':') and not line.endswith('?'):
            cleaned_lines.append(line)
        elif line and (line.endswith('.') or len(line) > 50):
            cleaned_lines.append(line)
    
    nudge = ' '.join(cleaned_lines).strip()
    
    # Validate nudge: must have actual content, not just headers
    if not nudge or len(nudge) < 20 or nudge.lower().startswith(('today\'s', 'your', 'nudge')):
        # Generate fallback nudge based on common patterns
        nudge = "Open your project and make one small commit. Set a 10-minute timer. Done? Yes/No"
    
    # Ensure nudge ends properly
    if not nudge.endswith(('.', '!', '?')):
        nudge = nudge.rstrip('.') + '.'
    
    # Break visualization into steps (split by sentence/paragraph)
    viz_sentences = [s.strip() for s in viz_text.split(". ") if len(s.strip()) > 20]
    if not viz_sentences:
        # Fallback steps
        viz_sentences = [
            "Close your eyes and take three deep breaths",
            "Feel your body relaxing with each exhale",
            "See yourself in your future, achieving your dream",
            "Notice the confidence and calm you feel",
            "Open your eyes, ready to take action"
        ]
    
    # Limit to 5 steps, each ~20 seconds
    steps = viz_sentences[:5]
    
    return {
        "nudge": nudge,
        "visualization": {
            "title": "Future Self Journey",
            "full_text": viz_text,
            "steps": steps,
            "duration_seconds": len(steps) * 20
        }
    }


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


@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Serve the frontend HTML at root URL."""
    # Try multiple paths to find index.html
    index_paths = [
        os.path.join(os.path.dirname(__file__), "static", "index.html"),  # Container: ./static/index.html
        os.path.join(os.path.dirname(__file__), "nudge-agent", "static", "index.html"),  # Local: nudge-agent/static/index.html
        os.path.join(os.path.dirname(__file__), "..", "nudge-agent", "static", "index.html"),  # Alternative
    ]
    
    index_path = None
    for path in index_paths:
        if os.path.exists(path):
            index_path = path
            break
    
    if index_path:
        print(f"✓ Serving index.html from: {index_path}")
        return FileResponse(index_path)
    else:
        # Fallback to API info if static file not found
        print("⚠ Warning: index.html not found, serving fallback")
        return HTMLResponse("""
        <html>
            <head>
                <title>Nudge Coach API</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 40px; text-align: center; }
                    a { color: #007AFF; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <h1>Nudge Coach API</h1>
                <p>Version 1.0.0</p>
                <p><a href="/docs">API Documentation</a></p>
                <p><a href="/api/v1/health">Health Check</a></p>
                <p style="color: #86868B; margin-top: 40px;">Static files not found. Check Dockerfile and file paths.</p>
            </body>
        </html>
        """)


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

