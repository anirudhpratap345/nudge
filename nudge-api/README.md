# 🚀 Nudge Coach API

**The single change that 10x's your AI coach.**

Replace your generic LLM + zero memory setup with this stack → Your coach suddenly feels like a real senior who actually knows the user.

## 🎯 What This Does

| Before (Generic LLM) | After (Nudge) |
|---------------------|---------------|
| "Take a deep breath" | "That stuck feeling at 5 PM hits hard. Set a 15-min timer for the Redis logic. Hit start now? Yes/No" |
| Zero recall | "You shipped that caching layer last week - don't let today reset the momentum" |
| Always high-energy therapy tone | Matches low energy: "Haan yaar… samajh raha hoon, yeh phase heavy lagta hai" |
| No accountability | Every reply ends with "Aaj 2 LC kar lega? Yes/No" |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Server                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │   ChromaDB  │    │    Redis    │    │   Groq /   │  │
│  │  (Memory)   │    │   (Cache)   │    │   Local    │  │
│  │             │    │             │    │    LLM     │  │
│  │  NV-Embed   │    │  Last 10    │    │            │  │
│  │  Retrieval  │    │  Messages   │    │  Llama 3.1 │  │
│  └─────────────┘    └─────────────┘    └────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone and setup
cd nudge-api
cp env.example .env

# 2. Add your Groq API key (free at console.groq.com)
# Edit .env: GROQ_API_KEY=your_key_here

# 3. Run
docker-compose up -d

# 4. Test
curl http://localhost:8000/api/v1/health
```

### Option 2: Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp env.example .env
# Edit .env with your GROQ_API_KEY

# 4. (Optional) Start Redis for message caching
docker run -d -p 6379:6379 redis:7-alpine

# 5. Run the server
python main.py
```

### Option 3: Railway/Render One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

Set these environment variables:
- `GROQ_API_KEY` - Your Groq API key
- `LLM_PROVIDER` - `groq`
- `EMBEDDING_DEVICE` - `cpu` (for cloud deployments)

## 📡 API Endpoints

### Chat (Main Endpoint)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "message": "I'\''m so burnt out. Did 6 hours of LeetCode and got nowhere."
  }'
```

**Response:**
```json
{
  "response": "Haan yaar... 6 hours of LC is brutal. Here's the thing - you're not stuck, you're saturated.\n\nEk kaam kar: Close LeetCode right now. Write down ONE pattern you understood today in 3 lines max.\n\nThat note becomes tomorrow's warmup. Did you close LC? Yes/No",
  "user_id": "user_123",
  "memories_used": 5,
  "timestamp": "2025-12-05T10:30:00Z"
}
```

### Store Memory

```bash
curl -X POST http://localhost:8000/api/v1/memory \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "content": "User shipped their first Redis caching feature to production",
    "memory_type": "win",
    "metadata": {"project": "PMArchitect"}
  }'
```

### Get User Memories

```bash
curl http://localhost:8000/api/v1/memory/user_123?query=recent+wins
```

### Delete User Data (GDPR)

```bash
curl -X DELETE http://localhost:8000/api/v1/memory/user_123
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` (recommended) or `local` |
| `GROQ_API_KEY` | - | Your Groq API key (free tier: 1M+ tokens/month) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `EMBEDDING_MODEL` | `nvidia/NV-Embed-v2` | Embedding model (#1 on MTEB) |
| `EMBEDDING_DEVICE` | `cuda` | `cuda` or `cpu` |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `MEMORY_TOP_K` | `8` | Number of memories to retrieve |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL (optional) |

### Using Fine-Tuned Model (Local)

If you've trained the Nudge LoRA adapter:

```bash
# 1. Place adapter in ./nudge-lora-adapter/

# 2. Update .env
LLM_PROVIDER=local
LOCAL_MODEL_PATH=./nudge-lora-adapter
LOCAL_BASE_MODEL=unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit

# 3. Install additional deps
pip install torch transformers peft bitsandbytes accelerate

# 4. Run (needs GPU with 8GB+ VRAM)
python main.py
```

## 🧠 The Magic: Memory Injection

Before every LLM call, we do this:

```python
# 1. Retrieve relevant memories using NV-Embed
memories = memory.retrieve_memories(
    user_id=user_id,
    query=user_message,
    n_results=8
)

# 2. Format as context
memory_context = memory.format_memory_context(memories)

# 3. Inject into prompt
response = llm.generate(
    user_message=user_message,
    memory_context=memory_context,  # ← This is the key
    conversation_history=history
)
```

**Result:** The coach remembers your Redis win from last week, your FAANG goal, your burnout phase, your family pressure - everything.

## 📊 Memory Types

Store different types of context for better retrieval:

| Type | Example | Use Case |
|------|---------|----------|
| `conversation` | "User said they're tired of DSA" | Auto-stored from chats |
| `goal` | "Crack Google by March 2026" | Long-term objectives |
| `win` | "Shipped first production feature" | Motivation fuel |
| `struggle` | "Dealing with imposter syndrome" | Context for empathy |
| `project` | "Building PMArchitect - PM interview prep tool" | Current work |

## 🎨 The Nudge Personality

The system prompt that makes it feel real:

```
You are "Nudge" — a sharp, caring, no-nonsense achievement coach 
for ambitious 20-somethings in India building careers and side projects.

You remember everything the user has ever told you.

You speak natural Indian English by default.
When the user sounds tired, low, or uses Hindi words, 
you naturally switch to light Hinglish.

You NEVER give generic advice like "take a deep breath".
Every suggestion is brutally specific and doable in ≤10 minutes.

You match the user's current energy first, then gently pull them forward.
You always end with a tiny accountability question (Yes/No or one number).
```

## 🔒 Security Notes

For production:
1. Set proper CORS origins (not `*`)
2. Add authentication (API keys, OAuth, etc.)
3. Rate limit endpoints
4. Use environment secrets for API keys
5. Enable HTTPS

## 📈 Scaling

- **Groq free tier**: 1M+ tokens/month, 500+ tok/s
- **ChromaDB**: Handles millions of embeddings
- **Redis**: Optional but improves response time
- **Oracle Cloud Always-Free**: 4 ARM cores, 24GB RAM for self-hosting

## 🤝 Integration with Existing App

Just replace your current LLM endpoint:

```python
# Before
response = openai.chat.completions.create(...)

# After
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={"user_id": user_id, "message": user_message}
)
nudge_response = response.json()["response"]
```

Everything else (UI, onboarding, notifications, database) stays the same.

---

**Result:** Retention and daily engagement explode because the coach finally feels like a real person who gives a damn and remembers your life.

