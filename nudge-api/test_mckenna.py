"""
Comprehensive Test Suite for Improved Advisor Nudge Endpoint
Tests 8 different user scenarios to validate McKenna-style coaching
"""
import requests
import json

# Change this to your actual deployed URL or localhost
BASE_URL = "http://localhost:8000"  # or "https://your-vercel-app.vercel.app"


def test_nudge(test_name, request_data):
    """Test the improved-advisor-nudge endpoint"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/improved-advisor-nudge",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n🎯 NUDGE:")
            print(f"{result['nudge']}")
            
            print(f"\n🌟 VISUALIZATION:")
            viz = result['visualization']
            print(f"Title: {viz.get('title', 'N/A')}")
            print(f"\n{viz.get('full_text', viz)}")
            
            if 'phases' in viz:
                print(f"\n📋 PHASES:")
                for phase, text in viz['phases'].items():
                    print(f"  {phase}: {text[:100]}...")
            
            print(f"\n📊 PERSONALITY INSIGHTS:")
            print(json.dumps(result.get('personality_insights', {}), indent=2))
            
            print(f"\n{'='*80}")
            print("✅ EVALUATION CHECKLIST:")
            
            # Check for McKenna-style language
            nudge_text = result['nudge'].lower()
            viz_text = viz.get('full_text', '').lower()
            combined_text = nudge_text + " " + viz_text
            
            checks = {
                "Uses sensory language (feel, see, hear, notice)": any(
                    word in combined_text for word in ['feel', 'see', 'hear', 'notice', 'sense']
                ),
                "Present-tense identity ('You ARE' not 'You will be')": any(
                    phrase in combined_text for phrase in ['you are', "you're", 'you have become']
                ),
                "Warm, gentle tone (not pushy/aggressive)": not any(
                    word in combined_text for word in ['must', 'have to', 'need to', 'should']
                ) or 'gentle' in combined_text or 'notice' in combined_text,
                "Specific micro-action (<10 min)": any(
                    word in nudge_text for word in ['open', 'write', 'solve', 'create', 'push', 'run']
                ),
                "Repetitive affirmations for rewiring": combined_text.count('you') >= 3 or combined_text.count('your') >= 2,
                "Different from generic AI advice": not any(
                    word in combined_text for word in ['remember', 'consider', 'think about', 'reflect on']
                )
            }
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"{status} {check}")
            
            print(f"{'='*80}\n")
            
        else:
            print(f"❌ ERROR {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ CONNECTION ERROR: Could not connect to {BASE_URL}")
        print("   Make sure the server is running: python -m uvicorn main:app --reload")
    except requests.exceptions.Timeout:
        print(f"❌ TIMEOUT: Request took longer than 30 seconds")
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# TEST CASES - 8 Different Scenarios
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MCKENNA ENDPOINT TEST SUITE")
    print("="*80)
    print(f"Testing endpoint: {BASE_URL}/api/v1/improved-advisor-nudge")
    print("="*80)

    # TEST 1: Low Energy + Procrastination (Most Common)
    test_nudge("Low Energy Procrastinator", {
        "user_id": "test_001_low_energy",
        "dream": "Launch my AI-powered SaaS product by March 2026",
        "progress": {
            "days_active": 15,
            "wins": 2,
            "struggles": ["procrastination", "overwhelm"]
        },
        "personality": {
            "energy_level": "low",
            "preferred_style": "gentle"
        }
    })

    # TEST 2: High Energy + Imposter Syndrome
    test_nudge("High Energy with Self-Doubt", {
        "user_id": "test_002_imposter",
        "dream": "Become a tech lead at a Series B startup by end of 2026",
        "progress": {
            "days_active": 60,
            "wins": 18,
            "struggles": ["imposter syndrome", "comparing to others"]
        },
        "personality": {
            "energy_level": "high",
            "preferred_style": "direct"
        }
    })

    # TEST 3: Complete Beginner (Vague Goals)
    test_nudge("Beginner with Vague Dreams", {
        "user_id": "test_003_beginner",
        "dream": "I want to be successful and make good money in tech",
        "progress": {
            "days_active": 3,
            "wins": 0,
            "struggles": ["lack of clarity", "don't know where to start"]
        },
        "personality": {
            "energy_level": "moderate",
            "preferred_style": "balanced"
        }
    })

    # TEST 4: Burned Out High Achiever
    test_nudge("Burned Out Overachiever", {
        "user_id": "test_004_burnout",
        "dream": "Build a sustainable career without burning out again",
        "progress": {
            "days_active": 120,
            "wins": 45,
            "struggles": ["burnout", "perfectionism", "work-life balance"]
        },
        "personality": {
            "energy_level": "low",
            "preferred_style": "gentle"
        }
    })

    # TEST 5: Career Transition (High Stakes)
    test_nudge("Career Switcher", {
        "user_id": "test_005_transition",
        "dream": "Transition from manual QA to AI/ML engineer by Dec 2026",
        "progress": {
            "days_active": 30,
            "wins": 8,
            "struggles": ["feeling behind", "age anxiety", "family pressure"]
        },
        "personality": {
            "energy_level": "moderate",
            "preferred_style": "direct"
        }
    })

    # TEST 6: Entrepreneur with Analysis Paralysis
    test_nudge("Analysis Paralysis Founder", {
        "user_id": "test_006_paralysis",
        "dream": "Launch my first paying customer for my coaching app",
        "progress": {
            "days_active": 90,
            "wins": 5,
            "struggles": ["overthinking", "perfectionism", "fear of shipping"]
        },
        "personality": {
            "energy_level": "moderate",
            "preferred_style": "gentle"
        }
    })

    # TEST 7: Highly Motivated (Everything Going Well)
    test_nudge("High Performer on Track", {
        "user_id": "test_007_momentum",
        "dream": "Get promoted to senior engineer and mentor a team",
        "progress": {
            "days_active": 75,
            "wins": 32,
            "struggles": []  # No major struggles
        },
        "personality": {
            "energy_level": "high",
            "preferred_style": "direct"
        }
    })

    # TEST 8: Specific Technical Goal
    test_nudge("Technical Skill Builder", {
        "user_id": "test_008_leetcode",
        "dream": "Master LeetCode and crack FAANG interviews by June 2026",
        "progress": {
            "days_active": 45,
            "wins": 15,
            "struggles": ["consistency", "hard problems frustration"]
        },
        "personality": {
            "energy_level": "moderate",
            "preferred_style": "balanced"
        }
    })

    print("\n" + "="*80)
    print("TESTING COMPLETE - Review each response above")
    print("="*80)
    print("\n💡 TIPS:")
    print("  - Check if responses use McKenna's hypnotic language patterns")
    print("  - Verify visualizations have 4 phases (GROUND, IMAGINE, EMBODY, COMMIT)")
    print("  - Ensure personality insights are extracted from memory")
    print("  - Compare responses across different energy levels and styles")
    print("="*80 + "\n")

