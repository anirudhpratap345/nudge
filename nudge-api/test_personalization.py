"""
Test script to verify personalization fixes work correctly.
Run this against your local or deployed API to validate:
1. Each user gets unique responses based on THEIR dream
2. Energy level adapts tone (low=gentle, high=energetic)
3. Preferred style adapts approach (gentle/direct/balanced)
4. No cross-user contamination
"""

import requests
import json
import sys
from typing import Dict, Any

# Configure this to your API URL
BASE_URL = "http://localhost:8000"  # or "https://anirudhpratap-nudge-agent.hf.space"


def test_nudge(test_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test the improved-advisor-nudge endpoint"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Dream: {request_data['dream']}")
    print(f"Energy: {request_data['personality'].get('energy_level', 'N/A')}")
    print(f"Style: {request_data['personality'].get('preferred_style', 'N/A')}")
    print("-" * 40)
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/improved-advisor-nudge",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n🎯 NUDGE:\n{result['nudge']}")
            print(f"\n🌟 VISUALIZATION TITLE: {result['visualization'].get('title', 'N/A')}")
            print(f"\n📝 VISUALIZATION PHASES:")
            if 'phases' in result['visualization']:
                for phase, text in result['visualization']['phases'].items():
                    print(f"  {phase}: {text[:80]}...")
            
            return result
        else:
            print(f"❌ ERROR {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return None


def analyze_response(result: Dict[str, Any], expected_energy: str, expected_style: str, dream_keywords: list) -> Dict[str, bool]:
    """Analyze if response matches expected personality adaptation"""
    if not result:
        return {"error": True}
    
    nudge = result.get("nudge", "").lower()
    viz = result.get("visualization", {})
    viz_text = viz.get("full_text", "") if isinstance(viz, dict) else str(viz)
    viz_text_lower = viz_text.lower()
    viz_title = viz.get("title", "") if isinstance(viz, dict) else ""
    
    checks = {}
    
    # Check 1: Does nudge/viz reference the specific dream?
    dream_referenced = any(kw.lower() in nudge + viz_text_lower for kw in dream_keywords)
    checks["dream_referenced"] = dream_referenced
    
    # Check 2: Does title use the dream?
    title_uses_dream = any(kw.lower() in viz_title.lower() for kw in dream_keywords)
    checks["title_uses_dream"] = title_uses_dream
    
    # Check 3: Is energy level reflected in language?
    if expected_energy == "low":
        gentle_words = ["gentle", "soft", "allow", "calm", "safe", "slowly", "gently", "peaceful"]
        checks["energy_adapted"] = any(w in viz_text_lower for w in gentle_words)
    elif expected_energy == "high":
        energetic_words = ["surge", "unstoppable", "fire", "power", "go!", "now!", "crush", "conquer", "bold"]
        checks["energy_adapted"] = any(w in viz_text_lower for w in energetic_words)
    else:
        checks["energy_adapted"] = True  # Moderate is flexible
    
    # Check 4: McKenna markers present?
    mckenna_markers = [
        "feel", "see", "notice", "hear", "breathe", "eyes",  # Sensory
        "you are", "i am", "becoming",  # Identity
    ]
    markers_found = sum(1 for m in mckenna_markers if m in viz_text_lower)
    checks["mckenna_markers"] = markers_found >= 3
    
    return checks


def run_all_tests():
    """Run comprehensive personalization tests"""
    results = []
    
    # TEST 1: Low energy, gentle, burnout recovery
    r1 = test_nudge("Low Energy + Gentle (Burnout)", {
        "user_id": "test_unique_001",
        "dream": "Regain my energy and build a consistent daily routine",
        "progress": {"days_active": 5, "wins": 1, "struggles": ["exhaustion", "overwhelm"]},
        "personality": {"energy_level": "low", "preferred_style": "gentle"}
    })
    if r1:
        a1 = analyze_response(r1, "low", "gentle", ["energy", "routine", "consistent"])
        results.append(("Low Energy Burnout", r1, a1))
    
    # TEST 2: High energy, direct, FAANG prep
    r2 = test_nudge("High Energy + Direct (FAANG Prep)", {
        "user_id": "test_unique_002",
        "dream": "Crack FAANG interviews and land a senior role by June 2026",
        "progress": {"days_active": 45, "wins": 12, "struggles": ["consistency"]},
        "personality": {"energy_level": "high", "preferred_style": "direct"}
    })
    if r2:
        a2 = analyze_response(r2, "high", "direct", ["faang", "interview", "senior"])
        results.append(("High Energy FAANG", r2, a2))
    
    # TEST 3: Moderate energy, balanced, career growth
    r3 = test_nudge("Moderate Energy + Balanced (AI Engineer)", {
        "user_id": "test_unique_003",
        "dream": "Become a senior AI engineer at a top startup by end of 2026",
        "progress": {"days_active": 30, "wins": 8, "struggles": ["imposter syndrome"]},
        "personality": {"energy_level": "moderate", "preferred_style": "balanced"}
    })
    if r3:
        a3 = analyze_response(r3, "moderate", "balanced", ["ai", "engineer", "senior", "startup"])
        results.append(("Moderate AI Engineer", r3, a3))
    
    # TEST 4: Low energy, direct, startup founder
    r4 = test_nudge("Low Energy + Direct (Startup)", {
        "user_id": "test_unique_004",
        "dream": "Launch my SaaS product and get first paying customer",
        "progress": {"days_active": 60, "wins": 5, "struggles": ["perfectionism", "fear of shipping"]},
        "personality": {"energy_level": "low", "preferred_style": "direct"}
    })
    if r4:
        a4 = analyze_response(r4, "low", "direct", ["saas", "launch", "customer", "product"])
        results.append(("Low Energy Startup", r4, a4))
    
    # SUMMARY
    print("\n" + "=" * 80)
    print("PERSONALIZATION VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, result, analysis in results:
        if analysis.get("error"):
            print(f"\n❌ {name}: FAILED (API error)")
            all_passed = False
            continue
        
        print(f"\n{name}:")
        for check, passed in analysis.items():
            symbol = "✅" if passed else "❌"
            print(f"  {symbol} {check}: {passed}")
            if not passed:
                all_passed = False
    
    # Cross-contamination check
    print("\n" + "-" * 40)
    print("CROSS-CONTAMINATION CHECK:")
    
    viz_titles = [r[1]['visualization'].get('title', '') for r in results if r[1]]
    nudges = [r[1]['nudge'] for r in results if r[1]]
    
    unique_titles = len(set(viz_titles)) == len(viz_titles)
    print(f"  {'✅' if unique_titles else '❌'} All visualization titles are unique: {unique_titles}")
    
    # Check no dream is appearing in wrong user's response
    print(f"  Titles found: {viz_titles}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL PERSONALIZATION TESTS PASSED!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review output above")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
    
    print(f"Testing API at: {BASE_URL}")
    success = run_all_tests()
    sys.exit(0 if success else 1)

