#!/usr/bin/env python3
"""
Streamlit UI Launcher for SOP QA Tool
This script properly launches the Streamlit app with correct module imports.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now we can import and run the streamlit app
if __name__ == "__main__":
    # Change to the project directory
    os.chdir(project_root)
    
    # Import streamlit and run the app
    import streamlit.web.cli as stcli
    
    # Set up the streamlit app path
    app_path = str(project_root / "sop_qa_tool" / "ui" / "streamlit_app.py")
    
    # Run streamlit with the correct module path
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--browser.gatherUsageStats=false"
    ]
    
    stcli.main()