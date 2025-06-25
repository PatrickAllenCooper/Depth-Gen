@echo off
echo ============================================
echo Depth-Gen Windows Setup Script
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Make sure you have CUDA installed and compatible PyTorch
    pause
    exit /b 1
)

REM Test CUDA availability
echo.
echo Testing CUDA availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA')"

REM Check for test video
if exist "test_depth_video.mp4" (
    echo.
    echo Test video found: test_depth_video.mp4
    echo.
    echo Setup complete! You can now run:
    echo   1. python test_video_depth_cuda.py    (Quick test)
    echo   2. python video_depth_processor.py test_depth_video.mp4    (Full processing)
    echo   3. uvicorn app.main:app --reload    (Start API server)
) else (
    echo.
    echo WARNING: test_depth_video.mp4 not found
    echo Please add your test video file to run the examples
)

echo.
echo ============================================
echo Setup completed successfully!
echo ============================================
pause 