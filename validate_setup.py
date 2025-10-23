#!/usr/bin/env python3
"""
Setup validation script for SOP QA Tool
Validates that the project setup is correct and all dependencies are available.
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.11+"""
    print("Checking Python version...")
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_virtual_environment():
    """Check if running in virtual environment"""
    print("Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        return True
    print("⚠️  Not running in virtual environment (recommended but not required)")
    return True


def check_project_structure():
    """Check if project structure is correct"""
    print("Checking project structure...")
    
    required_dirs = [
        "sop_qa_tool",
        "sop_qa_tool/config",
        "sop_qa_tool/models", 
        "sop_qa_tool/services",
        "sop_qa_tool/api",
        "sop_qa_tool/ui",
        "tests",
        "scripts"
    ]
    
    required_files = [
        "sop_qa_tool/__init__.py",
        "sop_qa_tool/config/__init__.py",
        "sop_qa_tool/config/settings.py",
        "requirements.txt",
        "bootstrap.ps1"
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ Missing directory: {dir_path}")
            all_good = False
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Missing file: {file_path}")
            all_good = False
        else:
            print(f"✅ File exists: {file_path}")
    
    return all_good


def check_configuration():
    """Check if configuration system works"""
    print("Checking configuration system...")
    
    try:
        from sop_qa_tool.config.settings import settings, validate_settings
        print("✅ Configuration module imported successfully")
        
        # Test settings access
        print(f"✅ Mode: {settings.mode}")
        print(f"✅ Chunk size: {settings.chunk_size}")
        print(f"✅ Max file size: {settings.max_file_size_mb}MB")
        
        # Validate settings
        if validate_settings():
            print("✅ Configuration validation passed")
            return True
        else:
            print("❌ Configuration validation failed")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import configuration: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def check_dependencies():
    """Check if key dependencies are available"""
    print("Checking key dependencies...")
    
    required_packages = [
        "pydantic",
        "pathlib",
        "os",
        "sys"
    ]
    
    optional_packages = [
        ("fastapi", "Web framework"),
        ("streamlit", "UI framework"),
        ("boto3", "AWS SDK"),
        ("sentence_transformers", "Local embeddings"),
        ("faiss", "Vector storage")
    ]
    
    all_good = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (required)")
            all_good = False
    
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} ({description})")
        except ImportError:
            print(f"⚠️  {package} ({description}) - install with: pip install -r requirements.txt")
    
    return all_good


def main():
    """Run all validation checks"""
    print("=== SOP QA Tool Setup Validation ===\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Project Structure", check_project_structure),
        ("Configuration System", check_configuration),
        ("Dependencies", check_dependencies)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n--- {check_name} ---")
        result = check_func()
        results.append((check_name, result))
    
    print("\n=== Validation Summary ===")
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All validation checks passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Review and modify .env file as needed")
        print("2. Install remaining dependencies: pip install -r requirements.txt")
        print("3. Run tests: python -m pytest tests/")
        return 0
    else:
        print("\n⚠️  Some validation checks failed. Please address the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())