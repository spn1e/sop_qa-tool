#!/usr/bin/env python3
"""
Entry point script for running the SOP Q&A Tool Streamlit UI.

This script starts both the FastAPI backend and Streamlit frontend.

Requirements: 5.1, 7.3
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sop_qa_tool.config.settings import get_settings


def start_api_server():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    
    # Change to project directory
    os.chdir(project_root)
    
    # Start uvicorn server
    cmd = [
        sys.executable, "-m", "uvicorn",
        "sop_qa_tool.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def start_streamlit_app():
    """Start the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    
    # Change to project directory
    os.chdir(project_root)
    
    # Start streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "sop_qa_tool/ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false"
    ]
    
    return subprocess.Popen(cmd)


def wait_for_api_ready(max_attempts=30, delay=1):
    """Wait for API server to be ready"""
    import requests
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"⏳ Waiting for API server... (attempt {attempt + 1}/{max_attempts})")
        time.sleep(delay)
    
    print("❌ API server failed to start within timeout period")
    return False


def main():
    """Main entry point"""
    print("🏭 SOP Q&A Tool - Starting Application")
    print("=" * 50)
    
    # Load settings
    settings = get_settings()
    print(f"📋 Operation Mode: {settings.mode.value}")
    
    # Create necessary directories
    settings.create_directories()
    
    # Start API server
    api_process = start_api_server()
    
    # Wait for API to be ready
    if not wait_for_api_ready():
        print("❌ Failed to start API server")
        api_process.terminate()
        return 1
    
    # Start Streamlit app
    streamlit_process = start_streamlit_app()
    
    print("\n🎉 Application started successfully!")
    print("📊 FastAPI Backend: http://localhost:8000")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🎨 Streamlit Frontend: http://localhost:8501")
    print("\n💡 Press Ctrl+C to stop both servers")
    
    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutting down servers...")
        
        # Terminate processes
        if streamlit_process.poll() is None:
            streamlit_process.terminate()
            print("✅ Streamlit server stopped")
        
        if api_process.poll() is None:
            api_process.terminate()
            print("✅ API server stopped")
        
        print("👋 Goodbye!")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Wait for processes to complete
        while True:
            # Check if processes are still running
            api_running = api_process.poll() is None
            streamlit_running = streamlit_process.poll() is None
            
            if not api_running:
                print("❌ API server stopped unexpectedly")
                if streamlit_running:
                    streamlit_process.terminate()
                break
            
            if not streamlit_running:
                print("❌ Streamlit server stopped unexpectedly")
                if api_running:
                    api_process.terminate()
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())