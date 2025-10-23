#!/usr/bin/env python3
"""
SOP QA Tool - Comprehensive Health Check

This script performs a comprehensive health check of the SOP QA Tool system
including all components, dependencies, and functionality.

Usage:
    python health_check.py              # Basic health check
    python health_check.py --detailed   # Detailed component testing
    python health_check.py --fix        # Attempt to fix common issues
"""

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        return False, f"Python 3.11+ required, found {version.major}.{version.minor}"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if all required dependencies are installed"""
    required_packages = [
        "fastapi",
        "streamlit", 
        "pydantic",
        "sentence_transformers",
        "faiss",
        "unstructured",
        "pytesseract",
        "httpx",
        "numpy",
        "pandas"
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing

def check_environment_config() -> Tuple[bool, str]:
    """Check environment configuration"""
    env_file = Path(".env")
    if not env_file.exists():
        return False, ".env file not found"
    
    try:
        from sop_qa_tool.config.settings import get_settings
        settings = get_settings()
        return True, f"Configuration loaded: {settings.mode} mode"
    except Exception as e:
        return False, f"Configuration error: {e}"

def check_data_directories() -> Tuple[bool, List[str]]:
    """Check if required data directories exist"""
    required_dirs = [
        "data",
        "data/raw_docs",
        "data/chunks",
        "data/faiss_index",
        "data/logs"
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
    
    return len(missing) == 0, missing

def check_services() -> Tuple[bool, Dict[str, str]]:
    """Check if core services can be imported and initialized"""
    services = {
        "document_ingestion": "sop_qa_tool.services.document_ingestion.DocumentIngestionService",
        "embedder": "sop_qa_tool.services.embedder.EmbeddingService", 
        "vectorstore": "sop_qa_tool.services.vectorstore.VectorStoreService",
        "text_chunker": "sop_qa_tool.services.text_chunker.TextChunker",
        "rag_chain": "sop_qa_tool.services.rag_chain.RAGChain"
    }
    
    results = {}
    all_ok = True
    
    for service_name, service_path in services.items():
        try:
            module_path, class_name = service_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            service_class = getattr(module, class_name)
            # Try to instantiate (this will test configuration)
            service_class()
            results[service_name] = "✅ OK"
        except Exception as e:
            results[service_name] = f"❌ Error: {str(e)[:50]}..."
            all_ok = False
    
    return all_ok, results

async def check_api_health() -> Tuple[bool, str]:
    """Check if API is running and healthy"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return True, f"API healthy: {data.get('status', 'unknown')}"
            else:
                return False, f"API returned {response.status_code}"
    except Exception as e:
        return False, f"API not accessible: {str(e)[:50]}..."

def check_ui_accessibility() -> Tuple[bool, str]:
    """Check if UI is accessible"""
    try:
        import requests
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            return True, "UI accessible"
        else:
            return False, f"UI returned {response.status_code}"
    except Exception as e:
        return False, f"UI not accessible: {str(e)[:50]}..."

async def test_document_processing() -> Tuple[bool, str]:
    """Test document processing pipeline"""
    try:
        from sop_qa_tool.services.text_chunker import TextChunker
        from sop_qa_tool.services.embedder import EmbeddingService
        
        # Test text chunking
        chunker = TextChunker()
        test_text = "This is a test document for the SOP QA Tool. It contains multiple sentences to test chunking."
        chunks = chunker.chunk_document(test_text, "test_doc")
        
        if not chunks:
            return False, "Text chunking failed - no chunks generated"
        
        # Test embedding generation
        embedder = EmbeddingService()
        chunk_texts = [chunk.chunk_text for chunk in chunks[:2]]  # Test first 2 chunks
        result = await embedder.embed_texts(chunk_texts)
        
        if result.embeddings.shape[0] != len(chunk_texts):
            return False, f"Embedding mismatch: expected {len(chunk_texts)}, got {result.embeddings.shape[0]}"
        
        return True, f"Processing OK: {len(chunks)} chunks, {result.dimensions}D embeddings"
        
    except Exception as e:
        return False, f"Processing error: {str(e)[:50]}..."

def fix_common_issues() -> List[str]:
    """Attempt to fix common issues"""
    fixes_applied = []
    
    # Create missing directories
    required_dirs = ["data", "data/raw_docs", "data/chunks", "data/faiss_index", "data/logs"]
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            fixes_applied.append(f"Created directory: {dir_path}")
    
    # Create .env from template if missing
    env_file = Path(".env")
    env_template = Path(".env.template")
    if not env_file.exists() and env_template.exists():
        env_file.write_text(env_template.read_text())
        fixes_applied.append("Created .env from template")
    
    return fixes_applied

async def run_health_check(detailed: bool = False, fix_issues: bool = False) -> bool:
    """Run comprehensive health check"""
    print("🏭 SOP QA Tool - Health Check")
    print("=" * 50)
    
    if fix_issues:
        print("🔧 Attempting to fix common issues...")
        fixes = fix_common_issues()
        for fix in fixes:
            print(f"  ✅ {fix}")
        if fixes:
            print()
    
    all_checks_passed = True
    
    # Basic checks
    print("📋 Basic System Checks")
    print("-" * 30)
    
    # Python version
    ok, msg = check_python_version()
    status = "✅" if ok else "❌"
    print(f"{status} Python Version: {msg}")
    if not ok:
        all_checks_passed = False
    
    # Dependencies
    ok, missing = check_dependencies()
    status = "✅" if ok else "❌"
    if ok:
        print(f"{status} Dependencies: All required packages installed")
    else:
        print(f"{status} Dependencies: Missing {len(missing)} packages: {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}")
        all_checks_passed = False
    
    # Environment config
    ok, msg = check_environment_config()
    status = "✅" if ok else "❌"
    print(f"{status} Configuration: {msg}")
    if not ok:
        all_checks_passed = False
    
    # Data directories
    ok, missing = check_data_directories()
    status = "✅" if ok else "❌"
    if ok:
        print(f"{status} Data Directories: All required directories exist")
    else:
        print(f"{status} Data Directories: Missing {len(missing)} directories")
        all_checks_passed = False
    
    if detailed:
        print("\n🔧 Detailed Component Checks")
        print("-" * 30)
        
        # Service initialization
        ok, results = check_services()
        print("Services:")
        for service, result in results.items():
            print(f"  {result} {service}")
        if not ok:
            all_checks_passed = False
        
        # Document processing test
        ok, msg = await test_document_processing()
        status = "✅" if ok else "❌"
        print(f"{status} Document Processing: {msg}")
        if not ok:
            all_checks_passed = False
        
        print("\n🌐 Service Accessibility")
        print("-" * 30)
        
        # API health
        ok, msg = await check_api_health()
        status = "✅" if ok else "❌"
        print(f"{status} API Service: {msg}")
        
        # UI accessibility
        ok, msg = check_ui_accessibility()
        status = "✅" if ok else "❌"
        print(f"{status} UI Service: {msg}")
    
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 All checks passed! System is healthy and ready.")
        print("\n🚀 Quick Start:")
        print("  • API: python -m uvicorn sop_qa_tool.api.main:app --reload")
        print("  • UI: streamlit run sop_qa_tool/ui/streamlit_app.py")
        print("  • Demo: python demo.py")
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
        print("\n🔧 Suggested fixes:")
        print("  • Install missing dependencies: pip install -r requirements.txt")
        print("  • Run with --fix flag: python health_check.py --fix")
        print("  • Check the troubleshooting guide: docs/TROUBLESHOOTING.md")
    
    return all_checks_passed

def main():
    """Main health check function"""
    parser = argparse.ArgumentParser(description="SOP QA Tool Health Check")
    parser.add_argument("--detailed", action="store_true", help="Run detailed component tests")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues")
    
    args = parser.parse_args()
    
    try:
        success = asyncio.run(run_health_check(args.detailed, args.fix))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Health check interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()