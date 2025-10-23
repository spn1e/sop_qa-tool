@echo off
echo Starting SOP QA Tool with Docker...
echo.

echo Building Docker images...
docker build -t sop-qa-tool -f Dockerfile.simple .

if %ERRORLEVEL% NEQ 0 (
    echo Docker build failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo Starting API container...
docker run -d --name sop-qa-api -p 8000:8000 -v "%cd%\data:/home/app/data" -e MODE=local sop-qa-tool

echo.
echo Starting UI container...
docker run -d --name sop-qa-ui -p 8501:8501 -v "%cd%\data:/home/app/data" --link sop-qa-api:api -e API_BASE_URL=http://api:8000 sop-qa-tool streamlit run sop_qa_tool/ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0

echo.
echo SOP QA Tool is starting up...
echo API: http://localhost:8000
echo UI: http://localhost:8501
echo.
echo Press any key to stop the containers...
pause

echo Stopping containers...
docker stop sop-qa-api sop-qa-ui
docker rm sop-qa-api sop-qa-ui

echo Done!
pause