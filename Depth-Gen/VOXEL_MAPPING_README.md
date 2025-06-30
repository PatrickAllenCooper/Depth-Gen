# Voxel Mapping from Depth Frames

This module extends the Depth-Gen project with the capability to convert depth frames into 3D voxel maps, providing a veridical interpretation of the depicted space with physical realism.

## Features

- **Depth to 3D Point Cloud Conversion**: Converts 2D depth maps to 3D point clouds using camera intrinsics
- **Voxel Grid Generation**: Creates voxel representations with configurable resolution
- **Multi-frame Integration**: Accumulates multiple depth frames into a single coherent voxel map
- **Physical Realism**: Proper scaling, coordinate systems, and physical units (meters)
- **Multiple Export Formats**: PLY meshes, point clouds, and visualizations
- **Confidence-based Weighting**: Uses depth confidence for robust voxel occupancy
- **Outlier Removal**: Statistical filtering for clean reconstructions

## Installation

Install the additional dependencies for voxel mapping:

```bash
pip install open3d trimesh scikit-image matplotlib scipy
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

### Simple Demo

Run the demonstration script to see voxel mapping in action:

```bash
python demo_voxel_mapping.py
```

This will:
1. Process depth frames from the `test_frames/` directory
2. Generate 3D voxel maps
3. Export PLY files viewable in any 3D software
4. Create visualizations

### Basic Usage

```python
from voxel_mapper import VoxelMapper, CameraIntrinsics, VoxelGridConfig

# Setup camera (assuming 60° field of view)
camera = CameraIntrinsics.from_fov(width=640, height=480, fov_degrees=60.0)

# Configure voxel grid
config = VoxelGridConfig(
    voxel_size=0.02,        # 2cm voxels
    depth_scale=255.0,      # For 8-bit depth maps
    world_scale=0.01,       # Convert to meters
    x_min=-3.0, x_max=3.0,  # 6m x 6m x 6m volume
    y_min=-3.0, y_max=3.0,
    z_min=0.0, z_max=6.0
)

# Create mapper
mapper = VoxelMapper(camera, config)

# Process depth frame
mapper.process_depth_frame("depth.png", "color.png")

# Export results
mapper.export_to_mesh("output.ply")
mapper.export_to_pointcloud("points.ply")
```

## Key Components

### 1. Camera Intrinsics

The `CameraIntrinsics` class defines the camera parameters needed for 3D reconstruction:

- `fx, fy`: Focal lengths in pixels
- `cx, cy`: Principal point (image center)
- `width, height`: Image dimensions

You can create intrinsics from field of view:
```python
camera = CameraIntrinsics.from_fov(width, height, fov_degrees=60.0)
```

### 2. Voxel Grid Configuration

The `VoxelGridConfig` class controls the voxel generation:

- `voxel_size`: Size of each voxel in meters (smaller = more detail)
- `depth_scale`: Conversion factor from depth values to meters
- `world_scale`: Additional scaling factor
- `x_min, x_max, y_min, y_max, z_min, z_max`: Bounding box in meters
- `occupancy_threshold`: Threshold for considering a voxel occupied
- `confidence_weight`: Use confidence-based averaging
- `outlier_removal`: Enable statistical outlier filtering

### 3. Depth to 3D Conversion

The conversion process:
1. **Back-projection**: Uses camera intrinsics to convert pixel coordinates + depth to 3D points
2. **Filtering**: Removes invalid depths and outliers
3. **Voxelization**: Discretizes 3D points into voxel grid
4. **Accumulation**: Combines multiple observations with confidence weighting

## Advanced Usage

### Multi-Frame Accumulation

Process multiple frames to build a more complete 3D model:

```python
# Process sequence of frames
for i, depth_path in enumerate(depth_frames):
    # Optional: provide camera pose for each frame
    pose = np.eye(4)  # Identity = no movement
    
    # Process frame
    mapper.process_depth_frame(depth_path, color_path, pose)

# Export accumulated result
mapper.export_to_mesh("accumulated_model.ply")
```

### Custom Camera Parameters

If you know your camera's intrinsic parameters:

```python
camera = CameraIntrinsics(
    fx=525.0,  # Focal length X
    fy=525.0,  # Focal length Y  
    cx=320.0,  # Principal point X
    cy=240.0,  # Principal point Y
    width=640,
    height=480
)
```

### Confidence Maps

The system automatically computes depth confidence from gradients. You can also provide custom confidence maps:

```python
points, colors, confidences = mapper.depth_to_point_cloud(
    depth_map, 
    color_image,
    depth_confidence=custom_confidence_map
)
```

## Output Formats

### PLY Files
- **Meshes**: Smooth surfaces using marching cubes or voxel cubes
- **Point Clouds**: Raw 3D points with optional colors
- Compatible with MeshLab, CloudCompare, Blender, etc.

### Visualizations
- 2D projections (top-down, side views)
- Depth map comparisons
- Matplotlib-based previews

### Raw Data
- NumPy arrays of voxel grids
- Save/load functionality for further processing

## Physical Units and Scaling

The system uses metric units (meters) throughout:

1. **Depth Values**: Raw depth map values (e.g., 0-255)
2. **Depth Scale**: Converts to real units (e.g., 255 → 1 meter)
3. **World Scale**: Additional scaling if needed
4. **Voxel Size**: Size of each voxel in meters

Example for typical depth maps:
- 8-bit depth (0-255): `depth_scale=255.0` for 0-1m range
- 16-bit depth (0-65535): `depth_scale=1000.0` for millimeters
- Adjust `world_scale` to match your scene scale

## Performance Considerations

- **Voxel Size**: Smaller voxels = more detail but higher memory usage
- **Grid Bounds**: Limit to area of interest to save memory
- **Outlier Removal**: Improves quality but adds processing time
- **Multi-threading**: Uses NumPy/OpenCV for efficient processing

## Viewing Results

The generated PLY files can be viewed with:
- **Windows**: 3D Viewer (built-in), Paint 3D
- **Cross-platform**: MeshLab, CloudCompare, Blender
- **Online**: Various web-based PLY viewers

## Troubleshooting

### Out of Memory
- Increase voxel size
- Reduce grid bounds
- Process fewer frames at once

### Poor Quality Results
- Check depth map quality
- Adjust `depth_scale` to match your depth encoding
- Enable outlier removal
- Tune confidence thresholds

### No Voxels Generated
- Verify depth values are in expected range
- Check camera intrinsics
- Ensure grid bounds contain the scene

## Example Results

The system can generate:
- Room-scale reconstructions from depth video
- Object models from turntable sequences  
- Scene understanding for robotics/AR
- Volumetric data for analysis

## Future Extensions

This voxel mapping module provides a foundation for:
- SLAM integration for camera tracking
- Texture mapping from color images
- Semantic voxel labeling
- Dynamic scene reconstruction
- Mesh optimization and simplification

## API Reference

See `voxel_mapper.py` for detailed API documentation and additional parameters. 