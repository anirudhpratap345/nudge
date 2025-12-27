"""
Test cases for the Improved Advisor Nudge endpoint
Tests the McKenna-style coaching with personalized nudges and visualizations
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_improved_nudge():
    """Test the McKenna-style nudge endpoint"""
    
    request_data = {
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
    }
    
    print("=" * 60)
    print("TESTING: /api/v1/improved-advisor-nudge")
    print("=" * 60)
    print(f"\n📤 REQUEST:")
    print(json.dumps(request_data, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/improved-advisor-nudge",
            json=request_data,
            timeout=30
        )
        
        print(f"\n📥 STATUS: {response.status_code}")
        
        if response.status_code != 200:
            print(f"\n❌ ERROR: {response.text}")
            return
        
        print("\n" + "=" * 60)
        print("✅ RESPONSE")
        print("=" * 60)
        
        result = response.json()
        
        print(f"\n🎯 NUDGE:")
        print("-" * 60)
        print(result['nudge'])
        
        print(f"\n🌟 VISUALIZATION:")
        print("-" * 60)
        viz = result['visualization']
        print(f"Title: {viz.get('title', 'N/A')}")
        print(f"Duration: {viz.get('duration_seconds', 0)} seconds")
        print(f"\nFull Text:")
        print(viz.get('full_text', 'N/A'))
        
        if 'phases' in viz:
            print(f"\n📋 PHASES:")
            for phase, text in viz['phases'].items():
                print(f"  {phase}: {text[:80]}...")
        
        if 'steps' in viz:
            print(f"\n📝 STEPS:")
            for i, step in enumerate(viz['steps'], 1):
                if isinstance(step, dict):
                    print(f"  {i}. ({step.get('duration_seconds', 0)}s) {step.get('text', '')[:80]}...")
                else:
                    print(f"  {i}. {str(step)[:80]}...")
        
        print(f"\n📊 PERSONALITY INSIGHTS:")
        print("-" * 60)
        print(json.dumps(result['personality_insights'], indent=2))
        
        print(f"\n🔊 TTS READY: {result.get('tts_ready', False)}")
        
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETE")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ ERROR: Could not connect to {BASE_URL}")
        print("   Make sure the server is running: python -m uvicorn main:app --reload")
    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: Request timed out after 30 seconds")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


def test_multiple_scenarios():
    """Test multiple user scenarios"""
    
    scenarios = [
        {
            "name": "High Energy, Direct Style",
            "data": {
                "user_id": "demo_user_002",
                "dream": "Launch my SaaS product and get 1000 paying customers by Q2 2026",
                "progress": {
                    "days_active": 90,
                    "wins": 25,
                    "struggles": ["focus_issues"]
                },
                "personality": {
                    "energy_level": "high",
                    "preferred_style": "direct"
                }
            }
        },
        {
            "name": "Low Energy, Gentle Style",
            "data": {
                "user_id": "demo_user_003",
                "dream": "Crack FAANG interviews and land a $200k+ offer by March 2026",
                "progress": {
                    "days_active": 30,
                    "wins": 5,
                    "struggles": ["procrastination", "overwhelm", "anxiety"]
                },
                "personality": {
                    "energy_level": "low",
                    "preferred_style": "gentle"
                }
            }
        },
        {
            "name": "New User (No History)",
            "data": {
                "user_id": "demo_user_004",
                "dream": "Build a successful AI startup",
                "progress": {},
                "personality": {}
            }
        }
    ]
    
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE SCENARIOS")
    print("=" * 60)
    
    for scenario in scenarios:
        print(f"\n\n📋 SCENARIO: {scenario['name']}")
        print("-" * 60)
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/improved-advisor-nudge",
                json=scenario['data'],
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Status: {response.status_code}")
                print(f"🎯 Nudge: {result['nudge'][:100]}...")
                print(f"🌟 Viz Title: {result['visualization'].get('title', 'N/A')}")
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"Error: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        test_improved_nudge()
        test_multiple_scenarios()
    else:
        test_improved_nudge()
        print("\n💡 Tip: Run with --all flag to test multiple scenarios:")
        print("   python test_mckenna_endpoint.py --all")

