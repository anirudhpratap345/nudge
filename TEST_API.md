# Nudge API Test Commands

## Test Chat Endpoint

```bash
curl -X POST https://nudge-blue.vercel.app/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_123",
    "message": "I want to become a senior AI engineer"
  }'
```

## Expected JSON Response

```json
{
  "response": "As you're becoming the AI engineer who ships daily, open LeetCode and solve problem #2389. Done? Yes/No",
  "user_id": "test_user_123",
  "memories_used": 0,
  "timestamp": "2025-01-09T18:30:00.000000"
}
```

## Test Health Endpoint

```bash
curl https://nudge-blue.vercel.app/api/v1/health
```

## Expected Response

```json
{
  "status": "healthy",
  "llm_provider": "groq",
  "memory_status": "lazy_load",
  "version": "1.0.0"
}
```

## PowerShell Test (Windows)

```powershell
Invoke-WebRequest -Uri "https://nudge-blue.vercel.app/api/v1/chat" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"user_id":"test_user","message":"I want to become a senior AI engineer"}' | 
  Select-Object StatusCode, Content
```

