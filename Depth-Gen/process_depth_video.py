#!/usr/bin/env python3
"""
Process depth test video into voxel map with robust error handling.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from voxel_mapper import VoxelMapper, CameraIntrinsics, VoxelGridConfig

def process_depth_video_to_voxels(video_path: str, output_dir: str, 
                                 skip_frames: int = 5, 
                                 max_frames: int = 100,
                                 voxel_size: float = 0.05):
    """
    Process depth video into voxel map.
    
    Args:
        video_path: Path to depth video
        output_dir: Output directory
        skip_frames: Process every Nth frame
        max_frames: Maximum frames to process
        voxel_size: Size of voxels in meters
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps}")
    
    # Check if side-by-side format
    is_side_by_side = width > height * 1.5
    if is_side_by_side:
        print("  - Format: Side-by-side (color | depth)")
        depth_width = width // 2
    else:
        print("  - Format: Depth only")
        depth_width = width
    
    # Setup camera intrinsics
    camera_intrinsics = CameraIntrinsics.from_fov(depth_width, height, fov_degrees=60.0)
    
    # Configure voxel grid based on analysis from single frame
    voxel_config = VoxelGridConfig(
        voxel_size=voxel_size,
        depth_scale=25.5,      # 8-bit depth, 255 = 10m
        world_scale=1.0,
        min_depth=0.5,
        max_depth=20.0,
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        z_min=0.0,
        z_max=20.0,
        occupancy_threshold=0.1,
        confidence_weight=True,
        outlier_removal=False  # Disable for speed on video
    )
    
    # Create voxel mapper
    mapper = VoxelMapper(camera_intrinsics, voxel_config)
    
    # Process frames
    frames_to_process = min(total_frames // skip_frames, max_frames)
    print(f"\nProcessing {frames_to_process} frames (every {skip_frames} frames)...")
    
    frame_count = 0
    processed_count = 0
    cumulative_transform = np.eye(4)
    
    # Directory for extracted frames (optional)
    frames_dir = output_path / "extracted_frames"
    frames_dir.mkdir(exist_ok=True)
    
    with tqdm(total=frames_to_process) as pbar:
        while processed_count < frames_to_process and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every skip_frames-th frame
            if frame_count % skip_frames == 0:
                try:
                    # Extract depth from frame
                    if is_side_by_side:
                        # Right half is depth
                        depth_frame = frame[:, width//2:]
                        color_frame = frame[:, :width//2]
                    else:
                        depth_frame = frame
                        color_frame = None
                    
                    # Convert to grayscale if needed
                    if len(depth_frame.shape) == 3:
                        depth_frame_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
                    else:
                        depth_frame_gray = depth_frame
                    
                    # Analyze depth values
                    valid_depths = depth_frame_gray[depth_frame_gray > 0]
                    if len(valid_depths) == 0:
                        print(f"Warning: Frame {frame_count} has no valid depth data, skipping")
                        frame_count += 1
                        continue
                    
                    # Save frames (optional, for debugging)
                    if processed_count < 10:  # Save first 10 frames
                        depth_path = frames_dir / f"depth_{processed_count:04d}.png"
                        cv2.imwrite(str(depth_path), depth_frame_gray)
                        
                        if color_frame is not None:
                            color_path = frames_dir / f"color_{processed_count:04d}.png"
                            cv2.imwrite(str(color_path), color_frame)
                    
                    # Simple camera motion (can be improved with feature tracking)
                    if processed_count > 0:
                        # Small random motion to simulate handheld camera
                        cumulative_transform[0, 3] += np.random.uniform(-0.01, 0.01)
                        cumulative_transform[1, 3] += np.random.uniform(-0.005, 0.005)
                    
                    # Process frame into voxel map
                    # Create temporary depth file for the mapper
                    temp_depth = output_path / "temp_depth.png"
                    cv2.imwrite(str(temp_depth), depth_frame_gray)
                    
                    # Process with current pose
                    stats = mapper.process_depth_frame(
                        str(temp_depth),
                        color_path=None,
                        pose=cumulative_transform.copy()
                    )
                    
                    # Clean up temp file
                    temp_depth.unlink()
                    
                    processed_count += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'points': stats['points_generated'],
                        'voxels': stats['voxels_updated'],
                        'total_voxels': mapper.stats['voxels_occupied']
                    })
                    
                    # Save intermediate results every 20 frames
                    if processed_count % 20 == 0:
                        intermediate_path = output_path / f"intermediate_{processed_count:04d}.ply"
                        mapper.export_to_pointcloud(str(intermediate_path))
                        
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    
            frame_count += 1
    
    cap.release()
    
    # Export final results
    print("\nExporting final voxel map...")
    
    # Get final statistics
    centers, occupancies, colors = mapper.get_occupied_voxels()
    print(f"Final voxel count: {len(centers):,}")
    
    # Save voxel grid data
    mapper.save_voxel_grid(str(output_path / "video_voxel_grid.npz"))
    
    # Export as point cloud
    mapper.export_to_pointcloud(str(output_path / "video_voxel_points.ply"))
    print(f"✓ Point cloud saved")
    
    # Export as mesh (if not too many voxels)
    if len(centers) < 100000:
        mapper.export_to_mesh(str(output_path / "video_voxel_mesh.ply"), 
                            use_marching_cubes=True)
        print(f"✓ Mesh saved")
    else:
        print(f"Skipping mesh export (too many voxels: {len(centers)})")
    
    # Create visualization
    print("\nCreating visualization...")
    viz_path = output_path / "video_voxel_visualization.png"
    mapper.visualize_voxels(str(viz_path), show=False)
    print(f"✓ Visualization saved")
    
    # Save metadata
    metadata = {
        'video_path': video_path,
        'frames_processed': processed_count,
        'skip_frames': skip_frames,
        'voxel_size': voxel_size,
        'total_voxels': len(centers),
        'video_properties': {
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'fps': fps
        },
        'stats': mapper.stats
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print final statistics
    mapper.print_stats()
    
    print(f"\n✓ Video processing complete!")
    print(f"✓ Results saved to: {output_path}")
    print(f"✓ Processed {processed_count} frames from {video_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process depth video to voxel map")
    parser.add_argument("video", nargs='?', default="test_depth_video.mp4",
                       help="Input video path (default: test_depth_video.mp4)")
    parser.add_argument("-o", "--output", default="depth_video_voxels",
                       help="Output directory (default: depth_video_voxels)")
    parser.add_argument("--skip-frames", type=int, default=10,
                       help="Process every Nth frame (default: 10)")
    parser.add_argument("--max-frames", type=int, default=100,
                       help="Maximum frames to process (default: 100)")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                       help="Voxel size in meters (default: 0.05)")
    
    args = parser.parse_args()
    
    process_depth_video_to_voxels(
        args.video,
        args.output,
        args.skip_frames,
        args.max_frames,
        args.voxel_size
    ) 