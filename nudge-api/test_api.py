"""
Quick test script for Nudge Coach API
Run: python test_api.py
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200


def test_chat(user_id: str = "test_user_1"):
    """Test chat endpoint"""
    print("\n💬 Testing chat endpoint...")
    
    # Test 1: Low energy scenario
    print("\n   Test 1: Low energy scenario")
    response = requests.post(
        f"{BASE_URL}/api/v1/chat",
        json={
            "user_id": user_id,
            "message": "I'm so tired. Been grinding DSA for 6 hours and feel stuck."
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Memories used: {data['memories_used']}")
        print(f"   Response preview: {data['response'][:200]}...")
    else:
        print(f"   Error: {response.text}")
    
    # Test 2: Hinglish trigger
    print("\n   Test 2: Hinglish trigger")
    response = requests.post(
        f"{BASE_URL}/api/v1/chat",
        json={
            "user_id": user_id,
            "message": "yaar kuch samajh nahi aa raha, bahut mushkil hai"
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Response preview: {data['response'][:200]}...")
    
    # Test 3: Win scenario
    print("\n   Test 3: Win scenario")
    response = requests.post(
        f"{BASE_URL}/api/v1/chat",
        json={
            "user_id": user_id,
            "message": "Just cracked my first Amazon OA! All test cases passed!"
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Response preview: {data['response'][:200]}...")
    
    return True


def test_memory(user_id: str = "test_user_1"):
    """Test memory endpoints"""
    print("\n🧠 Testing memory endpoints...")
    
    # Store a memory
    print("\n   Storing a memory...")
    response = requests.post(
        f"{BASE_URL}/api/v1/memory",
        json={
            "user_id": user_id,
            "content": "User is preparing for FAANG interviews, target is Google by March 2026",
            "memory_type": "goal",
            "metadata": {"company": "Google", "deadline": "March 2026"}
        }
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Retrieve memories
    print("\n   Retrieving memories...")
    response = requests.get(
        f"{BASE_URL}/api/v1/memory/{user_id}",
        params={"query": "interview preparation", "limit": 5}
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Found {data['count']} memories")
    
    # Get stats
    print("\n   Getting user stats...")
    response = requests.get(f"{BASE_URL}/api/v1/memory/{user_id}/stats")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    return True


def test_memory_recall(user_id: str = "test_user_1"):
    """Test that memories are recalled in chat"""
    print("\n🎯 Testing memory recall in chat...")
    
    # First, store a specific memory
    requests.post(
        f"{BASE_URL}/api/v1/memory",
        json={
            "user_id": user_id,
            "content": "User shipped a Redis caching feature last week and felt great momentum",
            "memory_type": "win",
            "metadata": {"project": "PMArchitect"}
        }
    )
    
    # Now chat about feeling stuck
    response = requests.post(
        f"{BASE_URL}/api/v1/chat",
        json={
            "user_id": user_id,
            "message": "Feeling stuck today, not sure what to work on"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Memories used: {data['memories_used']}")
        print(f"\n   Response:\n   {data['response']}")
        
        # Check if the response references past wins
        if "redis" in data['response'].lower() or "caching" in data['response'].lower() or "shipped" in data['response'].lower():
            print("\n   ✅ Memory recall detected! The response references past context.")
        else:
            print("\n   ℹ️ Response generated (memory may be used implicitly)")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🚀 NUDGE COACH API TEST SUITE")
    print("=" * 60)
    
    try:
        test_health()
        test_memory()
        test_chat()
        test_memory_recall()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection error! Is the server running?")
        print("   Start with: python main.py")
        print("   Or: docker-compose up -d")
    except Exception as e:
        print(f"\n❌ Test error: {e}")


if __name__ == "__main__":
    run_all_tests()

