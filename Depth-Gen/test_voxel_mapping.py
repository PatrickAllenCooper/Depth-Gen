#!/usr/bin/env python3
"""
Test script for voxel mapping from depth frames.
Demonstrates the conversion of depth maps to 3D voxel representations.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from voxel_mapper import VoxelMapper, CameraIntrinsics, VoxelGridConfig


def test_single_frame_voxel_mapping():
    """Test voxel mapping on a single depth frame."""
    print("=== Single Frame Voxel Mapping Test ===")
    
    # Setup paths
    test_frame_depth = "test_frames/frame_000_000000_depth.png"
    test_frame_color = "test_frames/frame_000_000000.png"
    output_dir = Path("voxel_test_output/single_frame")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load depth frame to get dimensions
    depth_map = cv2.imread(test_frame_depth, cv2.IMREAD_ANYDEPTH)
    if depth_map is None:
        depth_map = cv2.imread(test_frame_depth, cv2.IMREAD_GRAYSCALE)
    
    if depth_map is None:
        print(f"Error: Could not load depth frame from {test_frame_depth}")
        return
    
    height, width = depth_map.shape
    print(f"Depth map dimensions: {width}x{height}")
    
    # Create camera intrinsics (assuming 60° FOV)
    camera_intrinsics = CameraIntrinsics.from_fov(width, height, fov_degrees=60.0)
    
    # Configure voxel grid for single frame
    voxel_config = VoxelGridConfig(
        voxel_size=0.01,        # 1cm voxels for high detail
        depth_scale=255.0,      # Assuming 8-bit depth normalized to 0-255
        world_scale=0.01,       # Scale to realistic size (1 unit = 1cm)
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
    
    # Process single frame
    print("\nProcessing depth frame...")
    stats = mapper.process_depth_frame(test_frame_depth, test_frame_color)
    print(f"Generated {stats['points_generated']} points")
    print(f"Updated {stats['voxels_updated']} voxels")
    
    # Export results
    print("\nExporting results...")
    
    # Save as mesh
    mapper.export_to_mesh(str(output_dir / "single_frame_mesh.ply"), use_marching_cubes=True)
    mapper.export_to_pointcloud(str(output_dir / "single_frame_points.ply"))
    
    # Visualize
    mapper.visualize_voxels(str(output_dir / "single_frame_voxels.png"), show=False)
    
    # Print statistics
    mapper.print_stats()
    
    print(f"\n✓ Single frame test complete! Results in: {output_dir}")


def test_multi_frame_voxel_mapping():
    """Test voxel mapping on multiple depth frames with accumulation."""
    print("\n=== Multi-Frame Voxel Mapping Test ===")
    
    # Setup paths
    frame_dir = Path("test_frames")
    output_dir = Path("voxel_test_output/multi_frame")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all depth frames
    depth_frames = sorted(frame_dir.glob("frame_*_depth.png"))[:5]  # Use first 5 frames
    
    if not depth_frames:
        print("No depth frames found!")
        return
    
    print(f"Found {len(depth_frames)} depth frames to process")
    
    # Get dimensions from first frame
    first_depth = cv2.imread(str(depth_frames[0]), cv2.IMREAD_ANYDEPTH)
    if first_depth is None:
        first_depth = cv2.imread(str(depth_frames[0]), cv2.IMREAD_GRAYSCALE)
    
    height, width = first_depth.shape
    
    # Create camera intrinsics
    camera_intrinsics = CameraIntrinsics.from_fov(width, height, fov_degrees=60.0)
    
    # Configure voxel grid for multi-frame (larger bounds)
    voxel_config = VoxelGridConfig(
        voxel_size=0.02,        # 2cm voxels for balance of detail and memory
        depth_scale=255.0,
        world_scale=0.01,
        min_depth=0.1,
        max_depth=10.0,
        x_min=-3.0,
        x_max=3.0,
        y_min=-3.0,
        y_max=3.0,
        z_min=0.0,
        z_max=6.0,
        occupancy_threshold=0.4,
        confidence_weight=True,
        outlier_removal=True
    )
    
    # Create voxel mapper
    mapper = VoxelMapper(camera_intrinsics, voxel_config)
    
    # Process each frame
    print("\nProcessing frames...")
    for i, depth_path in enumerate(depth_frames):
        # Find corresponding color frame
        color_path = str(depth_path).replace("_depth.png", ".png")
        if not Path(color_path).exists():
            color_path = None
        
        # Simulate slight camera movement (optional)
        # In real scenarios, you would have actual camera poses
        pose = np.eye(4)
        if i > 0:
            # Small translation to simulate handheld camera movement
            pose[0, 3] = 0.05 * np.sin(i * 0.5)  # Small X movement
            pose[1, 3] = 0.03 * np.cos(i * 0.5)  # Small Y movement
        
        print(f"Processing frame {i+1}/{len(depth_frames)}: {depth_path.name}")
        stats = mapper.process_depth_frame(str(depth_path), color_path, pose)
        print(f"  Points: {stats['points_generated']}, Voxels updated: {stats['voxels_updated']}")
    
    # Export accumulated results
    print("\nExporting accumulated voxel map...")
    
    # Save voxel grid
    mapper.save_voxel_grid(str(output_dir / "accumulated_voxel_grid.npz"))
    
    # Export as mesh
    mapper.export_to_mesh(str(output_dir / "accumulated_mesh.ply"), use_marching_cubes=True)
    mapper.export_to_mesh(str(output_dir / "accumulated_voxels.ply"), use_marching_cubes=False)
    
    # Export as point cloud
    mapper.export_to_pointcloud(str(output_dir / "accumulated_points.ply"))
    
    # Visualize
    mapper.visualize_voxels(str(output_dir / "accumulated_voxels.png"), show=False)
    
    # Print final statistics
    mapper.print_stats()
    
    print(f"\n✓ Multi-frame test complete! Results in: {output_dir}")


def test_voxel_resolution_comparison():
    """Compare different voxel resolutions."""
    print("\n=== Voxel Resolution Comparison Test ===")
    
    # Setup
    test_frame_depth = "test_frames/frame_000_000000_depth.png"
    test_frame_color = "test_frames/frame_000_000000.png"
    output_dir = Path("voxel_test_output/resolution_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load depth frame
    depth_map = cv2.imread(test_frame_depth, cv2.IMREAD_ANYDEPTH)
    if depth_map is None:
        depth_map = cv2.imread(test_frame_depth, cv2.IMREAD_GRAYSCALE)
    
    height, width = depth_map.shape
    camera_intrinsics = CameraIntrinsics.from_fov(width, height, fov_degrees=60.0)
    
    # Test different voxel sizes
    voxel_sizes = [0.005, 0.01, 0.02, 0.05]  # 5mm to 5cm
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, voxel_size in enumerate(voxel_sizes):
        print(f"\nTesting voxel size: {voxel_size*100:.1f}cm")
        
        # Configure with different voxel size
        voxel_config = VoxelGridConfig(
            voxel_size=voxel_size,
            depth_scale=255.0,
            world_scale=0.01,
            x_min=-2.0, x_max=2.0,
            y_min=-2.0, y_max=2.0,
            z_min=0.0, z_max=4.0
        )
        
        # Create mapper and process
        mapper = VoxelMapper(camera_intrinsics, voxel_config)
        mapper.process_depth_frame(test_frame_depth, test_frame_color)
        
        # Get voxel data
        centers, occupancies, colors = mapper.get_occupied_voxels()
        
        # Plot on subplot
        ax = axes[i]
        if len(centers) > 0:
            # Sample if too many points
            if len(centers) > 5000:
                indices = np.random.choice(len(centers), 5000, replace=False)
                centers_plot = centers[indices]
                colors_plot = colors[indices] if np.any(colors > 0) else None
            else:
                centers_plot = centers
                colors_plot = colors if np.any(colors > 0) else None
            
            # 2D projection (top-down view)
            if colors_plot is not None and np.any(colors_plot > 0):
                scatter = ax.scatter(centers_plot[:, 0], centers_plot[:, 2], 
                                   c=colors_plot, s=10, alpha=0.6)
            else:
                scatter = ax.scatter(centers_plot[:, 0], centers_plot[:, 2], 
                                   c='blue', s=10, alpha=0.6)
        
        ax.set_title(f'Voxel Size: {voxel_size*100:.1f}cm\n{len(centers)} voxels')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Save individual mesh
        mesh_path = output_dir / f"mesh_voxel_{int(voxel_size*1000)}mm.ply"
        mapper.export_to_mesh(str(mesh_path), use_marching_cubes=True)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / "resolution_comparison.png"), dpi=150)
    plt.close()
    
    print(f"\n✓ Resolution comparison complete! Results in: {output_dir}")


def visualize_depth_to_3d_process():
    """Visualize the depth to 3D conversion process."""
    print("\n=== Depth to 3D Conversion Visualization ===")
    
    output_dir = Path("voxel_test_output/conversion_process")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load a depth frame
    depth_path = "test_frames/frame_000_000000_depth.png"
    color_path = "test_frames/frame_000_000000.png"
    
    depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_map is None:
        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    color_image = None
    if Path(color_path).exists():
        color_image = cv2.imread(color_path)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Original depth map
    ax1 = fig.add_subplot(131)
    ax1.imshow(depth_map, cmap='viridis')
    ax1.set_title('Original Depth Map')
    ax1.axis('off')
    
    # 2. Point cloud projection
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Create simple camera intrinsics
    h, w = depth_map.shape
    camera_intrinsics = CameraIntrinsics.from_fov(w, h, 60.0)
    
    # Convert to point cloud (simplified)
    voxel_config = VoxelGridConfig(depth_scale=255.0, world_scale=0.01)
    mapper = VoxelMapper(camera_intrinsics, voxel_config)
    points, colors_pc, _ = mapper.depth_to_point_cloud(depth_map, color_image)
    
    # Sample points for visualization
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points_vis = points[indices]
        colors_vis = colors_pc[indices] if colors_pc is not None else None
    else:
        points_vis = points
        colors_vis = colors_pc
    
    if colors_vis is not None:
        ax2.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                   c=colors_vis, s=1, alpha=0.5)
    else:
        ax2.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                   c=points_vis[:, 2], cmap='viridis', s=1, alpha=0.5)
    
    ax2.set_title('3D Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 3. Voxelized representation
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Process to voxels
    mapper.process_depth_frame(depth_path, color_path)
    centers, _, colors_vox = mapper.get_occupied_voxels()
    
    # Sample voxels
    if len(centers) > 5000:
        indices = np.random.choice(len(centers), 5000, replace=False)
        centers_vis = centers[indices]
        colors_vox_vis = colors_vox[indices] if np.any(colors_vox > 0) else None
    else:
        centers_vis = centers
        colors_vox_vis = colors_vox if np.any(colors_vox > 0) else None
    
    if colors_vox_vis is not None and np.any(colors_vox_vis > 0):
        ax3.scatter(centers_vis[:, 0], centers_vis[:, 1], centers_vis[:, 2],
                   c=colors_vox_vis, s=20, alpha=0.8, marker='s')
    else:
        ax3.scatter(centers_vis[:, 0], centers_vis[:, 1], centers_vis[:, 2],
                   c=centers_vis[:, 2], cmap='viridis', s=20, alpha=0.8, marker='s')
    
    ax3.set_title('Voxel Representation')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / "depth_to_3d_process.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Process visualization saved to: {output_dir}")


def main():
    """Run all voxel mapping tests."""
    print("="*50)
    print("VOXEL MAPPING TEST SUITE")
    print("="*50)
    
    # Check if test frames exist
    if not Path("test_frames").exists():
        print("Error: test_frames directory not found!")
        print("Please ensure you have depth frames in the test_frames directory.")
        return
    
    # Run tests
    try:
        # Test 1: Single frame
        test_single_frame_voxel_mapping()
        
        # Test 2: Multiple frames
        test_multi_frame_voxel_mapping()
        
        # Test 3: Resolution comparison
        test_voxel_resolution_comparison()
        
        # Test 4: Process visualization
        visualize_depth_to_3d_process()
        
        print("\n" + "="*50)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        print("\nVoxel mapping results are available in:")
        print("  - voxel_test_output/single_frame/")
        print("  - voxel_test_output/multi_frame/")
        print("  - voxel_test_output/resolution_comparison/")
        print("  - voxel_test_output/conversion_process/")
        
        print("\nYou can view the .ply files with:")
        print("  - MeshLab (free, cross-platform)")
        print("  - CloudCompare (free, cross-platform)")
        print("  - Blender (free, cross-platform)")
        print("  - Or any 3D viewer that supports PLY format")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 