#!/usr/bin/env python3
"""
Simple demonstration of voxel mapping from depth frames.
Run this to see voxel mapping in action!
"""

import os
import sys
from pathlib import Path

# Check dependencies
try:
    import numpy as np
    import cv2
    import open3d as o3d
    import trimesh
    import matplotlib.pyplot as plt
    from voxel_mapper import VoxelMapper, CameraIntrinsics, VoxelGridConfig
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nPlease install required packages:")
    print("pip install open3d trimesh scikit-image matplotlib scipy")
    sys.exit(1)


def demo_voxel_mapping():
    """Simple demonstration of voxel mapping."""
    print("=" * 60)
    print("DEPTH TO VOXEL MAPPING DEMONSTRATION")
    print("=" * 60)
    
    # Check for test frames
    test_frames_dir = Path("test_frames")
    if not test_frames_dir.exists():
        print("Error: test_frames directory not found!")
        return
    
    # Find depth frames
    depth_frames = sorted(test_frames_dir.glob("frame_*_depth.png"))
    if not depth_frames:
        print("Error: No depth frames found in test_frames/")
        return
    
    print(f"\nFound {len(depth_frames)} depth frames")
    
    # Use first depth frame for demo
    depth_path = depth_frames[0]
    color_path = Path(str(depth_path).replace("_depth.png", ".png"))
    
    print(f"\nProcessing: {depth_path.name}")
    
    # Load depth to get dimensions
    depth_map = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth_map is None:
        depth_map = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    
    height, width = depth_map.shape
    print(f"Depth map size: {width}x{height}")
    
    # Create output directory
    output_dir = Path("demo_voxel_output")
    output_dir.mkdir(exist_ok=True)
    
    # Setup voxel mapper with reasonable defaults
    print("\nSetting up voxel mapper...")
    camera_intrinsics = CameraIntrinsics.from_fov(width, height, fov_degrees=60.0)
    
    voxel_config = VoxelGridConfig(
        voxel_size=0.015,       # 1.5cm voxels
        depth_scale=255.0,      # 8-bit depth map
        world_scale=0.01,       # Scale to meters
        min_depth=0.1,
        max_depth=10.0,
        x_min=-2.5,
        x_max=2.5,
        y_min=-2.5,
        y_max=2.5,
        z_min=0.0,
        z_max=5.0
    )
    
    mapper = VoxelMapper(camera_intrinsics, voxel_config)
    
    # Process depth frame
    print("\nConverting depth to voxels...")
    color_path_str = str(color_path) if color_path.exists() else None
    stats = mapper.process_depth_frame(str(depth_path), color_path_str)
    
    print(f"✓ Generated {stats['points_generated']:,} 3D points")
    print(f"✓ Updated {stats['voxels_updated']:,} voxels")
    
    # Get voxel statistics
    centers, occupancies, colors = mapper.get_occupied_voxels()
    print(f"✓ Total occupied voxels: {len(centers):,}")
    
    # Export results
    print("\nExporting 3D models...")
    
    # Export as point cloud
    points_path = output_dir / "voxel_points.ply"
    mapper.export_to_pointcloud(str(points_path))
    print(f"✓ Point cloud saved: {points_path}")
    
    # Export as mesh
    mesh_path = output_dir / "voxel_mesh.ply"
    mapper.export_to_mesh(str(mesh_path), use_marching_cubes=True)
    print(f"✓ Mesh saved: {mesh_path}")
    
    # Create visualization
    print("\nCreating visualization...")
    viz_path = output_dir / "voxel_visualization.png"
    
    # Create a simple 2D projection visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show original depth map
    ax1.imshow(depth_map, cmap='viridis')
    ax1.set_title('Original Depth Map')
    ax1.axis('off')
    
    # Show top-down view of voxels
    if len(centers) > 0:
        # Sample voxels if too many
        if len(centers) > 10000:
            indices = np.random.choice(len(centers), 10000, replace=False)
            centers_plot = centers[indices]
            colors_plot = colors[indices] if np.any(colors > 0) else None
        else:
            centers_plot = centers
            colors_plot = colors if np.any(colors > 0) else None
        
        # Plot top-down view (X-Z plane)
        if colors_plot is not None and np.any(colors_plot > 0):
            scatter = ax2.scatter(centers_plot[:, 0], centers_plot[:, 2], 
                                c=colors_plot, s=5, alpha=0.6)
        else:
            # Use depth as color
            scatter = ax2.scatter(centers_plot[:, 0], centers_plot[:, 2], 
                                c=centers_plot[:, 2], cmap='viridis', s=5, alpha=0.6)
            plt.colorbar(scatter, ax=ax2, label='Depth (m)')
    
    ax2.set_title('Voxel Map (Top-Down View)')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (meters)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(viz_path), dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {viz_path}")
    
    # Process multiple frames if available
    if len(depth_frames) > 1:
        print("\n" + "-" * 60)
        print("MULTI-FRAME ACCUMULATION")
        print("-" * 60)
        
        # Reset mapper for multi-frame
        mapper = VoxelMapper(camera_intrinsics, voxel_config)
        
        # Process up to 5 frames
        num_frames = min(5, len(depth_frames))
        print(f"\nProcessing {num_frames} frames for accumulated map...")
        
        for i, depth_path in enumerate(depth_frames[:num_frames]):
            color_path = Path(str(depth_path).replace("_depth.png", ".png"))
            color_path_str = str(color_path) if color_path.exists() else None
            
            # Simple camera motion simulation (optional)
            pose = np.eye(4)
            if i > 0:
                # Slight translation to simulate movement
                pose[0, 3] = 0.05 * (i - num_frames/2)  # X translation
            
            print(f"  Frame {i+1}/{num_frames}: {depth_path.name}")
            stats = mapper.process_depth_frame(str(depth_path), color_path_str, pose)
        
        # Export accumulated results
        print("\nExporting accumulated voxel map...")
        
        mesh_path = output_dir / "accumulated_mesh.ply"
        mapper.export_to_mesh(str(mesh_path), use_marching_cubes=True)
        print(f"✓ Accumulated mesh saved: {mesh_path}")
        
        # Final statistics
        centers, _, _ = mapper.get_occupied_voxels()
        print(f"✓ Total voxels in accumulated map: {len(centers):,}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved in: {output_dir}/")
    print("\nYou can view the .ply files with:")
    print("  - MeshLab (meshlab.net)")
    print("  - CloudCompare (cloudcompare.org)")
    print("  - Blender (blender.org)")
    print("  - Windows 3D Viewer (built-in)")
    print("\nThe PNG visualization shows:")
    print("  - Left: Original depth map")
    print("  - Right: Top-down view of the generated voxel map")


if __name__ == "__main__":
    try:
        demo_voxel_mapping()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc() 