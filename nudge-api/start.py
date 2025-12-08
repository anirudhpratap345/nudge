#!/usr/bin/env python3
"""
Startup script for Nudge Coach API
Handles both development and production environments
"""
import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if required packages are installed"""
    required = ['fastapi', 'uvicorn', 'pydantic', 'faiss', 'sentence_transformers', 'groq']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Installing requirements...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed!")
    else:
        print("✅ All dependencies present")

def check_env():
    """Check environment configuration"""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    env_example = os.path.join(os.path.dirname(__file__), 'env.example')
    
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            import shutil
            shutil.copy(env_example, env_file)
            print("⚠️  Created .env from env.example - please add your GROQ_API_KEY!")
        else:
            print("⚠️  No .env file found. Using defaults (Mock LLM will be used)")
    else:
        print("✅ .env file found")
    
    # Check for Groq API key
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    if not os.getenv('GROQ_API_KEY'):
        print("⚠️  GROQ_API_KEY not set - Mock LLM will be used")
        print("   Get a free key at: https://console.groq.com")
    else:
        print("✅ GROQ_API_KEY configured")

def start_server(host='0.0.0.0', port=8000, reload=False, open_browser=True):
    """Start the FastAPI server"""
    import uvicorn
    
    print(f"\n🚀 Starting Nudge Coach API on http://{host}:{port}")
    print(f"   📚 API Docs: http://localhost:{port}/docs")
    print(f"   💬 Frontend: Open frontend/index.html in browser")
    print("\n" + "="*50)
    
    if open_browser:
        # Open API docs in browser after a short delay
        def open_docs():
            time.sleep(2)
            webbrowser.open(f'http://localhost:{port}/docs')
        
        import threading
        threading.Thread(target=open_docs, daemon=True).start()
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Start Nudge Coach API')
    parser.add_argument('--port', type=int, default=8000, help='Port to run on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser')
    parser.add_argument('--check-only', action='store_true', help='Only check dependencies')
    
    args = parser.parse_args()
    
    print("="*50)
    print("🎯 NUDGE COACH API STARTUP")
    print("="*50 + "\n")
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    check_dependencies()
    check_env()
    
    if args.check_only:
        print("\n✅ Dependency check complete!")
        return
    
    start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        open_browser=not args.no_browser
    )

if __name__ == "__main__":
    main()

