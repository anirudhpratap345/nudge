# McKenna-Style Coaching Endpoint

## Overview

The `/api/v1/improved-advisor-nudge` endpoint delivers Unlimits' core value proposition: personalized AI coaching powered by Paul McKenna's transformative techniques.

## Features

✅ **Micro-actions** (≤10 min, concrete, today)  
✅ **Hypnotic language** (sensory-rich, present-tense identity)  
✅ **Belief rewiring** (repetitive affirmations)  
✅ **Warm empathy** (understanding, never pushy)  
✅ **Deep personalization** (progress + personality + memory analysis)  
✅ **Structured visualizations** (4-phase guided sessions)

## Quick Start

### 1. Start the Server

```bash
cd nudge-api
python -m uvicorn main:app --reload
```

### 2. Test the Endpoint

```bash
python test_mckenna_endpoint.py
```

### 3. Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/improved-advisor-nudge" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user_001",
    "dream": "Become a senior AI engineer at a top startup by Dec 2026",
    "progress": {
      "days_active": 45,
      "wins": 12,
      "struggles": ["procrastination", "imposter syndrome"]
    },
    "personality": {
      "energy_level": "moderate",
      "preferred_style": "gentle"
    }
  }'
```

## Response Structure

```json
{
  "nudge": "Open LeetCode and solve problem #2389. Focus on the pattern, not perfection. Done? Yes/No",
  "visualization": {
    "title": "Journey to: Become a senior AI engineer...",
    "phases": {
      "GROUND": "Close your eyes. Take three deep breaths...",
      "IMAGINE": "See yourself achieving your dream...",
      "EMBODY": "You ARE the person who has achieved this...",
      "COMMIT": "Open your eyes. Take one small action..."
    },
    "steps": [
      {"text": "...", "duration_seconds": 15},
      {"text": "...", "duration_seconds": 30},
      {"text": "...", "duration_seconds": 30},
      {"text": "...", "duration_seconds": 15}
    ],
    "full_text": "...",
    "duration_seconds": 90
  },
  "personality_insights": {
    "energy": "moderate",
    "style": "gentle",
    "energy_pattern": "low",
    "struggles": ["procrastination", "overwhelm"]
  },
  "tts_ready": false
}
```

## Implementation Details

### System Prompt

Uses `MCKENNA_SYSTEM_PROMPT` from `config.py` with:
- Hypnotic language patterns
- Sensory-rich descriptions
- Present-tense identity affirmations
- Repetitive rewiring phrases

### Memory Analysis

- Retrieves up to 50 memories for deep analysis
- Extracts energy patterns, struggle themes, communication style
- Merges insights into personality traits

### Visualization Generation

- 4-phase structure (GROUND → IMAGINE → EMBODY → COMMIT)
- 90-second total duration
- Sensory-rich, transformative language

## Testing

Run the test suite:

```bash
# Single test
python test_mckenna_endpoint.py

# Multiple scenarios
python test_mckenna_endpoint.py --all
```

## Deployment

### Local
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Vercel
Already deployed at: `https://nudge-blue.vercel.app/api/v1/improved-advisor-nudge`

## Next Steps

- [ ] Add TTS integration (Piper/Coqui)
- [ ] Enhance personality analysis with LLM
- [ ] Add progress tracking metrics
- [ ] Implement visualization audio generation

## Pitch Script

```
"Hi, I'm [Your Name]. I analyzed Unlimits and identified a critical gap: 
your AI Coach delivers generic nudges without Paul McKenna's transformative language.

I built a prototype that fixes this. Let me show you..."

[DEMO]
1. Call /improved-advisor-nudge with sample user
2. Show the response:
   - Concrete micro-action (not generic advice)
   - Hypnotic visualization with sensory details
   - Deep personalization from memory

"Notice the language: 'Feel the calm confidence...', 'You ARE the engineer...'
This is McKenna's hypnotic rewiring, not typical AI coaching.

The tech: FastAPI, Groq (free tier), FAISS memory, production-deployed.
Time to implement: 10 hours.
Cost to run: $0."
```

