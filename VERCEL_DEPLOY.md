# Vercel Deployment Guide for Nudge Coach API

## ✅ Files Created

```
/
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI app (with GET / health check)
│   ├── config.py       # Updated for Vercel env vars (no .env)
│   ├── models.py       # Pydantic models
│   ├── llm.py          # Groq LLM integration
│   └── memory.py       # FAISS + Redis memory
├── vercel.json         # Vercel routing config
└── requirements.txt    # Python dependencies
```

## 🚀 Deployment Steps

### 1. Push to GitHub

```bash
git add .
git commit -m "Add Vercel deployment structure"
git push origin main
```

### 2. Connect to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click **"New Project"**
3. Import your GitHub repository
4. Vercel will auto-detect Python and use `vercel.json`

### 3. Set Environment Variables

In Vercel Dashboard → Project Settings → Environment Variables, add:

```
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.1-8b-instant
```

**Optional (for Redis/Upstash):**
```
REDIS_URL=redis://:password@host:port
REDIS_PASSWORD=your_password
REDIS_SSL=true
REDIS_ENABLED=true
```

**Optional (for custom settings):**
```
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=/tmp/chroma_db
MEMORY_TOP_K=8
MAX_NEW_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9
```

### 4. Deploy

Vercel will automatically:
- Install dependencies from `requirements.txt`
- Build using `@vercel/python`
- Route all requests to `api/main.py`

## 🧪 Test Your Deployment

### Health Check
```bash
curl https://your-project.vercel.app/
# Should return: {"status": "Nudge is live"}
```

### Chat Endpoint
```bash
curl -X POST https://your-project.vercel.app/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "I want to become a senior AI engineer at Google"
  }'
```

### API Docs
Visit: `https://your-project.vercel.app/docs`

## 📝 Important Notes

1. **No .env file needed** - Vercel uses environment variables directly
2. **FAISS storage** - Uses `/tmp/chroma_db` (ephemeral, resets on cold start)
3. **Redis** - Optional but recommended for production (use Upstash free tier)
4. **Cold starts** - First request may be slow (~5-10s) due to model loading
5. **Memory persistence** - FAISS data is lost on serverless function restart

## 🔧 Troubleshooting

### Build Fails
- Check `requirements.txt` has all dependencies
- Ensure Python version is compatible (Vercel uses 3.9+)

### Runtime Errors
- Check environment variables are set correctly
- View logs in Vercel Dashboard → Functions → Logs

### Slow Responses
- First request loads embedding model (expected)
- Subsequent requests are fast
- Consider using Redis for better caching

## 🎯 What Changed from Local Version

1. **config.py**: Uses `os.getenv()` instead of `.env` file
2. **main.py**: Added `GET /` route returning `{"status": "Nudge is live"}`
3. **vercel.json**: Routes all requests to `api/main.py`
4. **requirements.txt**: Removed `python-dotenv` (not needed)

Everything else is **identical** - zero logic changes! 🎉

