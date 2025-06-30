#!/usr/bin/env python3
"""
Process a single depth frame to voxel map with adjusted parameters.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from voxel_mapper import VoxelMapper, CameraIntrinsics, VoxelGridConfig

def analyze_depth_map(depth_map):
    """Analyze depth map to determine appropriate scaling."""
    valid_depths = depth_map[depth_map > 0]
    if len(valid_depths) == 0:
        return None
    
    print(f"\nDepth Map Analysis:")
    print(f"  - Min depth value: {valid_depths.min()}")
    print(f"  - Max depth value: {valid_depths.max()}")
    print(f"  - Mean depth value: {valid_depths.mean():.1f}")
    print(f"  - Std deviation: {valid_depths.std():.1f}")
    
    # Show depth histogram
    hist, bins = np.histogram(valid_depths, bins=50)
    print(f"  - Depth range spans: {valid_depths.min()} to {valid_depths.max()}")
    
    return {
        'min': valid_depths.min(),
        'max': valid_depths.max(),
        'mean': valid_depths.mean(),
        'std': valid_depths.std()
    }

def process_single_depth_frame(depth_path: str):
    """Process a single depth frame with adaptive scaling."""
    
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
    
    # Analyze depth map to determine scaling
    depth_stats = analyze_depth_map(depth_map)
    
    # Create output directory
    output_dir = Path("single_frame_voxel_output_adjusted")
    output_dir.mkdir(exist_ok=True)
    
    # Setup camera intrinsics (assuming 60° FOV)
    camera_intrinsics = CameraIntrinsics.from_fov(width, height, fov_degrees=60.0)
    
    # Configure voxel grid with adaptive scaling
    # Adjust depth_scale based on actual depth values
    depth_range = depth_stats['max'] - depth_stats['min']
    
    # Assume the depth map encodes distance in a normalized way
    # Try different scaling strategies
    if depth_stats['max'] <= 255:
        # 8-bit depth, assume max depth of 10m
        depth_scale = 25.5  # 255 / 10m
        world_scale = 1.0
    else:
        # 16-bit or other encoding
        depth_scale = depth_stats['max'] / 10.0  # Assume 10m max depth
        world_scale = 1.0
    
    print(f"\nUsing depth_scale: {depth_scale:.1f} (maps {depth_stats['max']} to {depth_stats['max']/depth_scale:.1f}m)")
    
    voxel_config = VoxelGridConfig(
        voxel_size=0.05,        # 5cm voxels (larger for better coverage)
        depth_scale=depth_scale,
        world_scale=world_scale,
        min_depth=0.5,          # Minimum 0.5m
        max_depth=20.0,         # Maximum 20m
        x_min=-10.0,            # Wider bounds
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        z_min=0.0,
        z_max=20.0,
        occupancy_threshold=0.1,  # Lower threshold
        confidence_weight=True,
        outlier_removal=False     # Disable initially to see all points
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
    
    # If still few voxels, print some debug info
    if len(centers) < 100 and stats['points_generated'] > 1000:
        print("\nDebug: Low voxel count detected!")
        # Get first few 3D points to check their range
        points, _, _ = mapper.depth_to_point_cloud(depth_map)
        if len(points) > 0:
            print(f"Sample 3D points (first 5):")
            for i in range(min(5, len(points))):
                print(f"  Point {i}: X={points[i,0]:.2f}, Y={points[i,1]:.2f}, Z={points[i,2]:.2f}")
            
            print(f"\n3D Point cloud statistics:")
            print(f"  X range: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
            print(f"  Y range: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
            print(f"  Z range: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    
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
    im = ax1.imshow(depth_map, cmap='viridis')
    ax1.set_title('Original Depth Map')
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    
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
            scatter = ax2.scatter(centers_plot[:, 0], centers_plot[:, 2], 
                       c=colors_plot, s=5, alpha=0.6)
        else:
            scatter = ax2.scatter(centers_plot[:, 0], centers_plot[:, 2], 
                       c=centers_plot[:, 2], cmap='viridis', s=5, alpha=0.6)
            plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
    
    ax2.set_title(f'Voxel Map - Top View ({len(centers)} voxels)')
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
            scatter = ax3.scatter(centers_plot[:, 2], centers_plot[:, 1], 
                       c=centers_plot[:, 2], cmap='viridis', s=5, alpha=0.6)
            plt.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)
    
    ax3.set_title('Voxel Map - Side View')
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