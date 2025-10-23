@echo off
echo Starting SOP QA Tool API Server...
call .\sop-qa-venv\Scripts\Activate.bat
python -m uvicorn sop_qa_tool.api.main:app --host 0.0.0.0 --port 8000
pause