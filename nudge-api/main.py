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
static_dir = os.path.join(os.path.dirname(__file__), "nudge-agent", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


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
        
        # 2. Build progress summary
        progress_summary = _build_progress_summary(request.progress)
        
        # 3. Extract/format personality
        personality_traits = _extract_personality(
            request.personality,
            memories
        )
        
        # 3.5. Get deep personality profile from memory
        try:
            personality_profile = memory.get_personality_profile(request.user_id)
            if personality_profile and personality_profile.get("status") != "new_user":
                # Merge profile insights into traits
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
        
        # 4. Build McKenna prompt
        from datetime import datetime
        try:
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            today_date = datetime.now(ist).strftime("%A, %B %d, %Y")
        except ImportError:
            # Fallback if pytz not installed
            today_date = datetime.now().strftime("%A, %B %d, %Y")
        
        mckenna_prompt = MCKENNA_SYSTEM_PROMPT.format(
            dream=request.dream,
            progress_summary=progress_summary,
            personality_traits=personality_str,
            memory_context=memory_context if memory_context else "(No memories yet - this is a new user)",
            today_date=today_date
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
                    temperature=0.4,  # Slightly higher for creative visualization
                    max_tokens=400,   # More tokens for visualization
                    top_p=0.8
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq API error in improved_advisor_nudge: {e}")
                response_text = "NUDGE: Open your project and make one small commit. VISUALIZATION: Close your eyes. Feel your breath slowing. See yourself as the engineer who ships daily."
        else:
            # Fallback: use regular generate method
            response_text = llm.generate(
                user_message=user_query,
                memory_context=memory_context,
                conversation_history=conversation_history
            )
        
        # 7. Parse into nudge + visualization
        parsed = _parse_mckenna_response(response_text)
        
        # 8. Generate structured visualization (enhanced version)
        try:
            structured_viz = generate_structured_visualization(
                dream=request.dream,
                personality=personality_traits,
                llm=llm
            )
            
            # Merge structured visualization into parsed response
            parsed["visualization"] = {
                "title": structured_viz["title"],
                "phases": structured_viz["phases"],
                "steps": structured_viz["steps"],
                "full_text": structured_viz["full_text"],
                "duration_seconds": structured_viz["total_duration"]
            }
        except Exception as e:
            logger.warning(f"Failed to generate structured visualization, using parsed version: {e}")
            # Continue with parsed visualization if structured generation fails
        
        # 9. Store interaction in memory
        try:
            memory.store_memory(
                user_id=request.user_id,
                content=f"Dream: {request.dream}. Nudge: {parsed['nudge']}",
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
    
    Creates a 90-second guided visualization with 4 distinct phases:
    1. GROUND (15s): Breath awareness, present moment
    2. IMAGINE (30s): Vivid sensory details of achieving the dream
    3. EMBODY (30s): Feel the identity shift - "I AM this person"
    4. COMMIT (15s): One small action to take today
    """
    personality_str = ", ".join([f"{k}: {v}" for k, v in personality.items()]) if personality else "exploring preferences"
    
    viz_prompt = f"""Create a 90-second guided visualization for this dream: {dream}

Structure it in 4 phases:
1. GROUND (15s): Breath awareness, present moment
2. IMAGINE (30s): Vivid sensory details of achieving the dream
3. EMBODY (30s): Feel the identity shift - "I AM this person"
4. COMMIT (15s): One small action to take today

Use McKenna's language: sensory-rich, present tense, repetitive affirmations.

Personality context: {personality_str}

Format each phase on a new line starting with the phase name (e.g., "GROUND: ...")."""

    # Use LLM to generate visualization
    system_prompt = "You are Paul McKenna's AI assistant, expert in hypnotic visualization. Create sensory-rich, transformative visualizations."
    
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
                temperature=0.5,  # Higher for creative visualization
                max_tokens=300,
                top_p=0.8
            )
            viz_response = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            # Fallback visualization
            viz_response = """GROUND: Close your eyes. Take three deep breaths. Feel your body relaxing with each exhale.
IMAGINE: See yourself achieving your dream. Notice the details around you. Hear the sounds. Feel the confidence.
EMBODY: You ARE the person who has achieved this. Feel this identity in your bones. This is who you are becoming.
COMMIT: Open your eyes. Take one small action today that moves you closer to this reality."""
    else:
        # Fallback: use regular generate
        viz_response = llm.generate(
            user_message=viz_prompt,
            memory_context="",
            conversation_history=""
        )
    
    # Parse phases
    phases = {}
    phase_order = ["GROUND", "IMAGINE", "EMBODY", "COMMIT"]
    
    for line in viz_response.split("\n"):
        line = line.strip()
        if not line:
            continue
        for phase in phase_order:
            if line.upper().startswith(phase):
                # Extract text after phase name
                phase_text = line.split(":", 1)[1].strip() if ":" in line else line.replace(phase, "").strip()
                phases[phase] = phase_text
                break
    
    # Fill in missing phases with fallbacks
    fallbacks = {
        "GROUND": "Close your eyes. Take three deep breaths. Feel your body relaxing with each exhale.",
        "IMAGINE": "See yourself achieving your dream. Notice the details around you. Hear the sounds. Feel the confidence.",
        "EMBODY": "You ARE the person who has achieved this. Feel this identity in your bones. This is who you are becoming.",
        "COMMIT": "Open your eyes. Take one small action today that moves you closer to this reality."
    }
    
    for phase in phase_order:
        if phase not in phases:
            phases[phase] = fallbacks[phase]
    
    # Create steps from phases
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
    static_dir = os.path.join(os.path.dirname(__file__), "nudge-agent", "static")
    index_path = os.path.join(static_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        # Fallback to API info if static file not found
        return HTMLResponse("""
        <html>
            <head><title>Nudge Coach API</title></head>
            <body>
                <h1>Nudge Coach API</h1>
                <p>Version 1.0.0</p>
                <p><a href="/docs">API Documentation</a></p>
                <p><a href="/api/v1/health">Health Check</a></p>
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

