@echo off
echo Starting SOP QA Tool Streamlit UI...
call .\sop-qa-venv\Scripts\Activate.bat
python -m streamlit run sop_qa_tool/ui/streamlit_app.py --server.port 8502
pause