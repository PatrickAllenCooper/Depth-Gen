#!/bin/bash

echo "========================================"
echo "Depth to Voxel Mapping Demo"
echo "========================================"
echo

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please create a virtual environment first:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if required packages are installed
if ! python -c "import open3d" 2>/dev/null; then
    echo "Installing voxel mapping dependencies..."
    pip install open3d trimesh scikit-image matplotlib scipy
fi

# Run the demo
echo
echo "Running voxel mapping demonstration..."
echo "This will convert depth frames to 3D voxel maps."
echo
python demo_voxel_mapping.py

echo
echo "Demo complete! Check the demo_voxel_output folder for results."
echo
echo "You can view the .ply files with:"
echo "- MeshLab (sudo apt install meshlab)"
echo "- CloudCompare (snap install cloudcompare)"
echo "- Blender (snap install blender --classic)"
echo 