#!/usr/bin/env python3
"""
Process a single depth frame to voxel map.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from voxel_mapper import VoxelMapper, CameraIntrinsics, VoxelGridConfig

def process_single_depth_frame(depth_path: str):
    """Process a single depth frame and generate voxel map."""
    
    print(f"Processing depth frame: {depth_path}")
    
    # Load depth map
    depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_map is None:
        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    if depth_map is None:
        print(f"Error: Could not load depth frame from {depth_path}")
        return
    
    height, width = depth_map.shape
    print(f"Depth map dimensions: {width}x{height}")
    
    # Create output directory
    output_dir = Path("single_frame_voxel_output")
    output_dir.mkdir(exist_ok=True)
    
    # Setup camera intrinsics (assuming 60° FOV)
    camera_intrinsics = CameraIntrinsics.from_fov(width, height, fov_degrees=60.0)
    
    # Configure voxel grid
    voxel_config = VoxelGridConfig(
        voxel_size=0.01,        # 1cm voxels for detail
        depth_scale=255.0,      # 8-bit depth map
        world_scale=0.01,       # Scale to realistic size
        min_depth=0.1,
        max_depth=10.0,
        x_min=-2.0,
        x_max=2.0,
        y_min=-2.0,
        y_max=2.0,
        z_min=0.0,
        z_max=4.0,
        occupancy_threshold=0.3,
        confidence_weight=True,
        outlier_removal=True
    )
    
    # Create voxel mapper
    mapper = VoxelMapper(camera_intrinsics, voxel_config)
    
    # Check for corresponding color image
    color_path = depth_path.replace("_depth.png", ".png")
    if not Path(color_path).exists():
        color_path = None
        print("No color image found, using depth only")
    else:
        print(f"Found color image: {color_path}")
    
    # Process the frame
    print("\nConverting depth to voxels...")
    stats = mapper.process_depth_frame(depth_path, color_path)
    
    print(f"✓ Generated {stats['points_generated']:,} 3D points")
    print(f"✓ Updated {stats['voxels_updated']:,} voxels")
    
    # Get voxel statistics
    centers, occupancies, colors = mapper.get_occupied_voxels()
    print(f"✓ Total occupied voxels: {len(centers):,}")
    
    # Export results
    print("\nExporting 3D models...")
    
    # Export as point cloud
    points_path = output_dir / f"{Path(depth_path).stem}_points.ply"
    mapper.export_to_pointcloud(str(points_path))
    print(f"✓ Point cloud saved: {points_path}")
    
    # Export as smooth mesh
    mesh_path = output_dir / f"{Path(depth_path).stem}_mesh.ply"
    mapper.export_to_mesh(str(mesh_path), use_marching_cubes=True)
    print(f"✓ Mesh saved: {mesh_path}")
    
    # Export as voxel cubes
    voxel_path = output_dir / f"{Path(depth_path).stem}_voxels.ply"
    mapper.export_to_mesh(str(voxel_path), use_marching_cubes=False)
    print(f"✓ Voxel cubes saved: {voxel_path}")
    
    # Create visualization
    print("\nCreating visualization...")
    viz_path = output_dir / f"{Path(depth_path).stem}_visualization.png"
    
    # Create visualization
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Original depth map
    ax1 = fig.add_subplot(131)
    ax1.imshow(depth_map, cmap='viridis')
    ax1.set_title('Original Depth Map')
    ax1.axis('off')
    
    # 2. Top-down view
    ax2 = fig.add_subplot(132)
    if len(centers) > 0:
        # Sample if too many
        sample_size = min(10000, len(centers))
        if len(centers) > sample_size:
            indices = np.random.choice(len(centers), sample_size, replace=False)
            centers_plot = centers[indices]
            colors_plot = colors[indices] if np.any(colors > 0) else None
        else:
            centers_plot = centers
            colors_plot = colors if np.any(colors > 0) else None
        
        if colors_plot is not None and np.any(colors_plot > 0):
            ax2.scatter(centers_plot[:, 0], centers_plot[:, 2], 
                       c=colors_plot, s=5, alpha=0.6)
        else:
            ax2.scatter(centers_plot[:, 0], centers_plot[:, 2], 
                       c=centers_plot[:, 2], cmap='viridis', s=5, alpha=0.6)
    
    ax2.set_title('Voxel Map (Top View)')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (meters)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. Side view
    ax3 = fig.add_subplot(133)
    if len(centers) > 0:
        if colors_plot is not None and np.any(colors_plot > 0):
            ax3.scatter(centers_plot[:, 2], centers_plot[:, 1], 
                       c=colors_plot, s=5, alpha=0.6)
        else:
            ax3.scatter(centers_plot[:, 2], centers_plot[:, 1], 
                       c=centers_plot[:, 2], cmap='viridis', s=5, alpha=0.6)
    
    ax3.set_title('Voxel Map (Side View)')
    ax3.set_xlabel('Z (meters)')
    ax3.set_ylabel('Y (meters)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Invert Y to match image coordinates
    
    plt.tight_layout()
    plt.savefig(str(viz_path), dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {viz_path}")
    plt.close()
    
    # Save voxel grid data
    grid_path = output_dir / f"{Path(depth_path).stem}_voxel_grid.npz"
    mapper.save_voxel_grid(str(grid_path))
    print(f"✓ Voxel grid data saved: {grid_path}")
    
    # Print summary
    mapper.print_stats()
    
    print(f"\n✓ Processing complete!")
    print(f"✓ All results saved to: {output_dir}/")
    print("\nYou can view the .ply files with:")
    print("  - Windows 3D Viewer (double-click)")
    print("  - MeshLab, CloudCompare, or Blender")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        depth_frame = sys.argv[1]
    else:
        # Default to the frame mentioned
        depth_frame = "test_frames/frame_002_015592_depth.png"
    
    process_single_depth_frame(depth_frame) 