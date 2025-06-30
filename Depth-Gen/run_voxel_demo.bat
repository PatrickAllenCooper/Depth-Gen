@echo off
echo ========================================
echo Depth to Voxel Mapping Demo
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup_windows.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate

REM Check if required packages are installed
python -c "import open3d" 2>nul
if errorlevel 1 (
    echo Installing voxel mapping dependencies...
    pip install open3d trimesh scikit-image matplotlib scipy
)

REM Run the demo
echo.
echo Running voxel mapping demonstration...
echo This will convert depth frames to 3D voxel maps.
echo.
python demo_voxel_mapping.py

echo.
echo Demo complete! Check the demo_voxel_output folder for results.
echo.
echo You can view the .ply files with:
echo - Windows 3D Viewer (double-click the files)
echo - MeshLab (free download from meshlab.net)
echo - Blender (free download from blender.org)
echo.
pause 