"""
Test script to verify context awareness fixes:
1. Indecision/exploration detection
2. Context shift handling
3. Category-specific nudges
4. Memory contamination prevention

Run: python test_context_awareness.py [BASE_URL]
Default: http://localhost:8000
"""

import requests
import json
import sys
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"


def test_nudge(test_name: str, request_data: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, bool]:
    """Test the improved-advisor-nudge endpoint with expectations"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Dream: {request_data['dream']}")
    print(f"Expected state: {expected.get('state', 'N/A')}")
    print(f"Expected title pattern: {expected.get('title_contains', 'N/A')}")
    print("-" * 40)
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/improved-advisor-nudge",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ ERROR {response.status_code}: {response.text[:200]}")
            return {"error": True}
        
        result = response.json()
        
        # Extract response parts
        nudge = result.get("nudge", "")
        viz = result.get("visualization", {})
        viz_title = viz.get("title", "") if isinstance(viz, dict) else ""
        
        print(f"\n🎯 NUDGE: {nudge[:150]}...")
        print(f"\n📌 VISUALIZATION TITLE: {viz_title}")
        
        # Run checks
        checks = {}
        
        # Check 1: Title matches expected pattern
        title_pattern = expected.get("title_contains", "")
        if title_pattern:
            checks["title_correct"] = title_pattern.lower() in viz_title.lower()
            print(f"\n{'✅' if checks['title_correct'] else '❌'} Title contains '{title_pattern}': {checks['title_correct']}")
        
        # Check 2: Nudge is NOT generic "make a commit"
        generic_patterns = ["make one commit", "make a commit", "ship something small", "open your project and make"]
        is_generic = any(p in nudge.lower() for p in generic_patterns)
        checks["not_generic"] = not is_generic
        print(f"{'✅' if checks['not_generic'] else '❌'} Nudge is not generic: {checks['not_generic']}")
        
        # Check 3: If indecision expected, nudge should be comparison-focused
        if expected.get("state") == "indecision":
            comparison_words = ["compare", "comparison", "vs", "pros", "cons", "list", "column", "matrix", "options"]
            has_comparison = any(w in nudge.lower() for w in comparison_words)
            checks["handles_indecision"] = has_comparison
            print(f"{'✅' if has_comparison else '❌'} Nudge handles indecision: {has_comparison}")
        
        # Check 4: Category-specific nudge keywords
        if expected.get("nudge_keywords"):
            keywords = expected["nudge_keywords"]
            has_keywords = any(kw.lower() in nudge.lower() for kw in keywords)
            checks["category_specific"] = has_keywords
            print(f"{'✅' if has_keywords else '❌'} Contains category keywords {keywords}: {has_keywords}")
        
        # Check 5: Title does NOT contain wrong dream (memory contamination)
        if expected.get("should_not_contain"):
            contaminated = expected["should_not_contain"].lower() in viz_title.lower()
            checks["no_contamination"] = not contaminated
            print(f"{'✅' if checks['no_contamination'] else '❌'} No contamination from '{expected['should_not_contain']}': {checks['no_contamination']}")
        
        return checks
        
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return {"error": True}


def run_all_tests():
    """Run comprehensive context awareness tests"""
    all_results = []
    
    # =========================================================================
    # TEST GROUP 1: INDECISION DETECTION
    # =========================================================================
    print("\n\n" + "="*80)
    print("GROUP 1: INDECISION DETECTION")
    print("="*80)
    
    # Test 1.1: Clear "or" indecision
    r1 = test_nudge(
        "Indecision: Meta SWE vs Quant Dev (using 'or')",
        {
            "user_id": "ctx_test_001",
            "dream": "Should I become a software engineer at Meta or a quant developer?",
            "progress": {"days_active": 10, "wins": 2, "struggles": []},
            "personality": {"energy_level": "moderate", "preferred_style": "balanced"}
        },
        {
            "state": "indecision",
            "title_contains": "Exploring",
            "nudge_keywords": ["compare", "comparison", "pros", "cons", "list", "vs"]
        }
    )
    all_results.append(("Indecision: or", r1))
    
    # Test 1.2: "Also thinking about" indecision
    r2 = test_nudge(
        "Indecision: Also thinking about (follow-up)",
        {
            "user_id": "ctx_test_002",
            "dream": "I want to get into Meta, but also thinking about grad school",
            "progress": {"days_active": 30, "wins": 5, "struggles": ["clarity"]},
            "personality": {"energy_level": "moderate", "preferred_style": "balanced"}
        },
        {
            "state": "indecision",
            "title_contains": "Exploring",
            "nudge_keywords": ["compare", "pros", "cons", "options", "list"]
        }
    )
    all_results.append(("Indecision: also", r2))
    
    # Test 1.3: Multiple options with "vs"
    r3 = test_nudge(
        "Indecision: Triple options (FAANG vs Startup vs Grad School)",
        {
            "user_id": "ctx_test_003",
            "dream": "FAANG vs startup vs grad school - which path should I take?",
            "progress": {"days_active": 5, "wins": 0, "struggles": ["indecision"]},
            "personality": {"energy_level": "low", "preferred_style": "gentle"}
        },
        {
            "state": "indecision",
            "title_contains": "Exploring",
            "nudge_keywords": ["compare", "decision", "options", "matrix", "pros"]
        }
    )
    all_results.append(("Indecision: triple options", r3))
    
    # =========================================================================
    # TEST GROUP 2: CATEGORY-SPECIFIC NUDGES
    # =========================================================================
    print("\n\n" + "="*80)
    print("GROUP 2: CATEGORY-SPECIFIC NUDGES (Not Generic)")
    print("="*80)
    
    # Test 2.1: FAANG specific
    r4 = test_nudge(
        "Category: FAANG Interview Prep",
        {
            "user_id": "ctx_test_004",
            "dream": "Get into Meta as a software engineer",
            "progress": {"days_active": 45, "wins": 10, "struggles": []},
            "personality": {"energy_level": "high", "preferred_style": "direct"}
        },
        {
            "state": "single_goal",
            "title_contains": "Journey to",
            "nudge_keywords": ["leetcode", "problem", "system design", "interview", "coding"]
        }
    )
    all_results.append(("Category: FAANG", r4))
    
    # Test 2.2: Startup specific
    r5 = test_nudge(
        "Category: Startup Launch",
        {
            "user_id": "ctx_test_005",
            "dream": "Launch my SaaS product and get first paying customer",
            "progress": {"days_active": 60, "wins": 8, "struggles": ["shipping"]},
            "personality": {"energy_level": "moderate", "preferred_style": "direct"}
        },
        {
            "state": "single_goal",
            "title_contains": "Journey to",
            "nudge_keywords": ["ship", "launch", "feature", "customer", "mvp", "product", "tweet"]
        }
    )
    all_results.append(("Category: Startup", r5))
    
    # Test 2.3: Career transition specific
    r6 = test_nudge(
        "Category: Career Transition",
        {
            "user_id": "ctx_test_006",
            "dream": "Transition from QA to ML engineer",
            "progress": {"days_active": 20, "wins": 3, "struggles": ["skills gap"]},
            "personality": {"energy_level": "moderate", "preferred_style": "balanced"}
        },
        {
            "state": "single_goal",
            "title_contains": "Journey to",
            "nudge_keywords": ["linkedin", "skills", "course", "learn", "role", "profile", "transition"]
        }
    )
    all_results.append(("Category: Transition", r6))
    
    # Test 2.4: Burnout recovery
    r7 = test_nudge(
        "Category: Burnout Recovery",
        {
            "user_id": "ctx_test_007",
            "dream": "I'm completely burned out and need to recover my energy",
            "progress": {"days_active": 100, "wins": 30, "struggles": ["burnout"]},
            "personality": {"energy_level": "low", "preferred_style": "gentle"}
        },
        {
            "state": "single_goal",
            "title_contains": "Journey to",
            "nudge_keywords": ["rest", "step away", "breath", "small", "tiny", "walk", "break", "timer"]
        }
    )
    all_results.append(("Category: Burnout", r7))
    
    # Test 2.5: Quant specific
    r8 = test_nudge(
        "Category: Quant Developer",
        {
            "user_id": "ctx_test_008",
            "dream": "Become a quantitative developer at a hedge fund",
            "progress": {"days_active": 30, "wins": 5, "struggles": []},
            "personality": {"energy_level": "high", "preferred_style": "direct"}
        },
        {
            "state": "single_goal",
            "title_contains": "Journey to",
            "nudge_keywords": ["python", "pandas", "quant", "math", "finance", "algo", "data"]
        }
    )
    all_results.append(("Category: Quant", r8))
    
    # =========================================================================
    # TEST GROUP 3: MEMORY CONTAMINATION PREVENTION
    # =========================================================================
    print("\n\n" + "="*80)
    print("GROUP 3: NO MEMORY CONTAMINATION")
    print("="*80)
    
    # Test 3.1: First request establishes context
    test_nudge(
        "Memory Test Setup: First goal (SaaS launch)",
        {
            "user_id": "ctx_test_memory_001",
            "dream": "Launch my AI-powered SaaS and get first customer",
            "progress": {"days_active": 30, "wins": 5, "struggles": []},
            "personality": {"energy_level": "moderate", "preferred_style": "balanced"}
        },
        {"title_contains": "SaaS"}
    )
    
    # Test 3.2: Different user, completely different goal - should NOT be contaminated
    r9 = test_nudge(
        "Memory Test: Different user, different goal (FAANG)",
        {
            "user_id": "ctx_test_memory_002",  # Different user!
            "dream": "Crack Google interviews and become SWE there",
            "progress": {"days_active": 45, "wins": 10, "struggles": []},
            "personality": {"energy_level": "high", "preferred_style": "direct"}
        },
        {
            "title_contains": "Google",
            "should_not_contain": "SaaS"  # Should NOT contain previous user's goal
        }
    )
    all_results.append(("No contamination: different users", r9))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n\n" + "="*80)
    print("CONTEXT AWARENESS TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, checks in all_results:
        if checks.get("error"):
            print(f"\n❌ {name}: FAILED (API error)")
            all_passed = False
            continue
        
        test_passed = all(checks.values())
        symbol = "✅" if test_passed else "❌"
        print(f"\n{symbol} {name}:")
        for check_name, passed in checks.items():
            check_symbol = "  ✓" if passed else "  ✗"
            print(f"   {check_symbol} {check_name}")
        
        if not test_passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL CONTEXT AWARENESS TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - Review output above")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
    
    print(f"Testing Context Awareness at: {BASE_URL}")
    print("Testing fixes for:")
    print("  1. Indecision detection (or, also, vs)")
    print("  2. Category-specific nudges (not generic)")
    print("  3. Memory contamination prevention")
    print("-" * 40)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)

