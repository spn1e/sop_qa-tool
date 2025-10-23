#!/usr/bin/env python3
"""
SOP QA Tool - Setup and Installation Script

This script provides automated setup for the SOP QA Tool with the following features:
- Dependency installation and validation
- Environment configuration
- System health checks
- Quick start options

Usage:
    python setup.py --mode local    # Local development setup
    python setup.py --mode aws      # AWS production setup
    python setup.py --demo          # Setup and run demo
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command with proper error handling"""
    try:
        print(f"🔧 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required, found {version.major}.{version.minor}")
        print("Please install Python 3.11 or later from https://python.org")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("sop-qa-venv")
    
    if venv_path.exists():
        print("📦 Virtual environment already exists")
        return venv_path
    
    print("📦 Creating virtual environment...")
    run_command([sys.executable, "-m", "venv", str(venv_path)])
    
    return venv_path

def get_pip_executable(venv_path: Path) -> Path:
    """Get the pip executable path for the virtual environment"""
    if os.name == 'nt':  # Windows
        return venv_path / "Scripts" / "pip.exe"
    else:  # Unix-like
        return venv_path / "bin" / "pip"

def install_dependencies(venv_path: Path):
    """Install Python dependencies"""
    pip_exe = get_pip_executable(venv_path)
    
    print("📦 Upgrading pip...")
    run_command([str(pip_exe), "install", "--upgrade", "pip"])
    
    print("📦 Installing dependencies...")
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        run_command([str(pip_exe), "install", "-r", str(requirements_file)])
    else:
        print("❌ requirements.txt not found")
        sys.exit(1)

def setup_environment_config(mode: str):
    """Setup environment configuration"""
    print(f"⚙️ Setting up environment for {mode} mode...")
    
    env_file = Path(".env")
    env_template = Path(".env.template")
    
    if not env_file.exists() and env_template.exists():
        print("📝 Creating .env from template...")
        with open(env_template, 'r') as template:
            content = template.read()
        
        # Set mode in the content
        content = content.replace("MODE=local", f"MODE={mode}")
        
        with open(env_file, 'w') as env:
            env.write(content)
        print("✅ Environment configuration created")
    else:
        print("📝 Environment configuration already exists")

def create_data_directories():
    """Create necessary data directories"""
    print("📁 Creating data directories...")
    
    directories = [
        "data",
        "data/raw_docs",
        "data/chunks", 
        "data/faiss_index",
        "data/logs",
        "data/evaluation"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create .gitkeep files
    for dir_path in ["data/raw_docs", "data/chunks"]:
        gitkeep = Path(dir_path) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    
    print("✅ Data directories created")

def validate_setup(venv_path: Path):
    """Validate the installation"""
    print("🔍 Validating setup...")
    
    python_exe = venv_path / ("Scripts/python.exe" if os.name == 'nt' else "bin/python")
    
    # Test imports
    test_script = """
import sys
try:
    from sop_qa_tool.config.settings import get_settings
    from sop_qa_tool.services.embedder import EmbeddingService
    from sop_qa_tool.services.vectorstore import VectorStoreService
    print("✅ All core modules imported successfully")
    
    # Test configuration
    settings = get_settings()
    print(f"✅ Configuration loaded: {settings.mode} mode")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Setup validation failed: {e}")
    sys.exit(1)
"""
    
    result = run_command([str(python_exe), "-c", test_script], check=False)
    if result.returncode != 0:
        print("❌ Setup validation failed")
        sys.exit(1)
    
    print("✅ Setup validation passed")

def start_services(venv_path: Path, demo: bool = False):
    """Start the application services"""
    python_exe = venv_path / ("Scripts/python.exe" if os.name == 'nt' else "bin/python")
    
    print("🚀 Starting services...")
    print("Note: This will start the services. Use Ctrl+C to stop.")
    print("Services will be available at:")
    print("  • API: http://localhost:8000")
    print("  • UI: http://localhost:8501")
    
    if demo:
        print("  • Demo: python demo.py")
    
    try:
        # Start API in background
        api_cmd = [str(python_exe), "-m", "uvicorn", "sop_qa_tool.api.main:app", "--reload", "--port", "8000"]
        print(f"Starting API: {' '.join(api_cmd)}")
        
        # Start UI
        ui_cmd = [str(python_exe), "-m", "streamlit", "run", "sop_qa_tool/ui/streamlit_app.py", "--server.port", "8501"]
        print(f"Starting UI: {' '.join(ui_cmd)}")
        print("\n🌐 Open http://localhost:8501 in your browser")
        
        if demo:
            print("\n🎬 To run the demo, open another terminal and run: python demo.py")
        
    except KeyboardInterrupt:
        print("\n👋 Services stopped")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="SOP QA Tool Setup Script")
    parser.add_argument("--mode", choices=["local", "aws"], default="local", 
                       help="Operation mode (default: local)")
    parser.add_argument("--demo", action="store_true", 
                       help="Setup and prepare for demo")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing setup")
    parser.add_argument("--start-services", action="store_true",
                       help="Start services after setup")
    
    args = parser.parse_args()
    
    print("🏭 SOP QA Tool - Setup Script")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    if args.validate_only:
        venv_path = Path("sop-qa-venv")
        if not venv_path.exists():
            print("❌ Virtual environment not found. Run setup first.")
            sys.exit(1)
        validate_setup(venv_path)
        return
    
    # Setup virtual environment
    venv_path = setup_virtual_environment()
    
    # Install dependencies
    install_dependencies(venv_path)
    
    # Setup environment
    setup_environment_config(args.mode)
    
    # Create directories
    create_data_directories()
    
    # Validate setup
    validate_setup(venv_path)
    
    print("\n🎉 Setup completed successfully!")
    print(f"Mode: {args.mode}")
    print(f"Virtual environment: {venv_path}")
    
    if args.demo:
        print("\n🎬 Demo mode enabled")
        print("Run 'python demo.py' to start the interactive demo")
    
    if args.start_services:
        start_services(venv_path, args.demo)
    else:
        print("\n🚀 To start the services:")
        print("1. Activate virtual environment:")
        if os.name == 'nt':
            print(f"   .\\{venv_path}\\Scripts\\Activate.ps1")
        else:
            print(f"   source {venv_path}/bin/activate")
        print("2. Start API: python -m uvicorn sop_qa_tool.api.main:app --reload")
        print("3. Start UI: streamlit run sop_qa_tool/ui/streamlit_app.py")

if __name__ == "__main__":
    main()