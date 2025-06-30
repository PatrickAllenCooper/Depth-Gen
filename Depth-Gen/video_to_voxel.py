#!/usr/bin/env python3
"""
Video to Voxel Mapping Pipeline
Processes depth video output into 3D voxel reconstructions.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from voxel_mapper import VoxelMapper, CameraIntrinsics, VoxelGridConfig


class VideoToVoxelProcessor:
    """Process depth videos into voxel maps with camera tracking."""
    
    def __init__(self, camera_intrinsics: CameraIntrinsics, voxel_config: VoxelGridConfig):
        self.camera_intrinsics = camera_intrinsics
        self.voxel_config = voxel_config
        self.mapper = VoxelMapper(camera_intrinsics, voxel_config)
        
    def extract_frames_from_video(self, video_path: str, output_dir: Path, 
                                 skip_frames: int = 1, max_frames: int = None):
        """
        Extract frames from depth video.
        
        Args:
            video_path: Path to depth video
            output_dir: Directory to save extracted frames
            skip_frames: Process every Nth frame (1 = all frames)
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frame paths
        """
        output_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} frames at {fps:.1f} FPS")
        
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        with tqdm(total=min(total_frames, max_frames or total_frames), 
                  desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % skip_frames == 0:
                    # Extract depth channel (assuming side-by-side format)
                    height, width = frame.shape[:2]
                    
                    # Check if side-by-side format
                    if width > height * 1.5:  # Likely side-by-side
                        # Extract right half (depth)
                        depth_frame = frame[:, width//2:]
                    else:
                        # Assume entire frame is depth
                        depth_frame = frame
                    
                    # Convert to grayscale if needed
                    if len(depth_frame.shape) == 3:
                        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Save frame
                    frame_path = output_dir / f"depth_{extracted_count:06d}.png"
                    cv2.imwrite(str(frame_path), depth_frame)
                    frame_paths.append(frame_path)
                    
                    extracted_count += 1
                    pbar.update(1)
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
        
        cap.release()
        print(f"Extracted {extracted_count} frames")
        return frame_paths
    
    def estimate_camera_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Estimate camera motion between frames using feature matching.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            4x4 transformation matrix
        """
        # Simple identity transform for now
        # In a full implementation, you would use:
        # - Feature detection (SIFT, ORB, etc.)
        # - Feature matching
        # - Essential matrix estimation
        # - Pose recovery
        
        # For demonstration, simulate small camera movement
        transform = np.eye(4)
        
        # Small random translation to simulate handheld movement
        transform[0, 3] = np.random.uniform(-0.02, 0.02)  # X
        transform[1, 3] = np.random.uniform(-0.01, 0.01)  # Y
        transform[2, 3] = np.random.uniform(-0.01, 0.01)  # Z
        
        return transform
    
    def process_video(self, video_path: str, output_dir: str,
                     skip_frames: int = 5, max_frames: int = 100,
                     use_temporal_consistency: bool = True):
        """
        Process depth video into voxel map.
        
        Args:
            video_path: Path to depth video
            output_dir: Output directory for results
            skip_frames: Process every Nth frame
            max_frames: Maximum frames to process
            use_temporal_consistency: Apply temporal filtering
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Extract frames directory
        frames_dir = output_path / "extracted_frames"
        
        print(f"Processing video: {video_path}")
        
        # Extract frames from video
        frame_paths = self.extract_frames_from_video(
            video_path, frames_dir, skip_frames, max_frames
        )
        
        if not frame_paths:
            print("No frames extracted!")
            return
        
        # Process frames into voxel map
        print(f"\nBuilding voxel map from {len(frame_paths)} frames...")
        
        cumulative_transform = np.eye(4)
        prev_frame = None
        
        for i, frame_path in enumerate(tqdm(frame_paths, desc="Processing frames")):
            # Load depth frame
            depth_frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            
            # Estimate camera motion (simplified)
            if i > 0 and prev_frame is not None:
                # Estimate relative transform
                relative_transform = self.estimate_camera_motion(prev_frame, depth_frame)
                cumulative_transform = cumulative_transform @ relative_transform
            
            # Process frame with pose
            self.mapper.process_depth_frame(
                str(frame_path), 
                color_path=None,
                pose=cumulative_transform.copy()
            )
            
            prev_frame = depth_frame
            
            # Save intermediate results every 20 frames
            if (i + 1) % 20 == 0:
                intermediate_path = output_path / f"intermediate_{i+1:04d}.ply"
                self.mapper.export_to_mesh(str(intermediate_path))
        
        # Export final results
        print("\nExporting final voxel map...")
        
        # Save voxel grid data
        self.mapper.save_voxel_grid(str(output_path / "voxel_grid.npz"))
        
        # Export meshes
        self.mapper.export_to_mesh(str(output_path / "voxel_mesh_smooth.ply"), 
                                  use_marching_cubes=True)
        self.mapper.export_to_mesh(str(output_path / "voxel_mesh_cubes.ply"), 
                                  use_marching_cubes=False)
        
        # Export point cloud
        self.mapper.export_to_pointcloud(str(output_path / "voxel_points.ply"))
        
        # Generate visualization
        self.mapper.visualize_voxels(str(output_path / "voxel_visualization.png"), 
                                    show=False)
        
        # Save metadata
        metadata = {
            'video_path': video_path,
            'frames_processed': len(frame_paths),
            'skip_frames': skip_frames,
            'voxel_size': self.voxel_config.voxel_size,
            'camera_fov': 60.0,  # Assumed
            'stats': self.mapper.stats
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print statistics
        self.mapper.print_stats()
        
        print(f"\n✓ Video processing complete!")
        print(f"✓ Results saved to: {output_path}")
        
        # Clean up extracted frames if desired
        cleanup = input("\nDelete extracted frames? (y/N): ").lower() == 'y'
        if cleanup:
            import shutil
            shutil.rmtree(frames_dir)
            print("Cleaned up extracted frames")


def main():
    parser = argparse.ArgumentParser(
        description="Convert depth video to 3D voxel map"
    )
    parser.add_argument("video", help="Input depth video path")
    parser.add_argument("-o", "--output", default="voxel_output",
                       help="Output directory (default: voxel_output)")
    parser.add_argument("--skip-frames", type=int, default=5,
                       help="Process every Nth frame (default: 5)")
    parser.add_argument("--max-frames", type=int, default=100,
                       help="Maximum frames to process (default: 100)")
    parser.add_argument("--voxel-size", type=float, default=0.02,
                       help="Voxel size in meters (default: 0.02)")
    parser.add_argument("--fov", type=float, default=60.0,
                       help="Camera field of view in degrees (default: 60)")
    parser.add_argument("--depth-scale", type=float, default=255.0,
                       help="Depth value to meters scale (default: 255)")
    parser.add_argument("--world-scale", type=float, default=0.01,
                       help="Additional world scaling (default: 0.01)")
    
    # Voxel grid bounds
    parser.add_argument("--x-bounds", nargs=2, type=float, default=[-5.0, 5.0],
                       help="X axis bounds in meters (default: -5 5)")
    parser.add_argument("--y-bounds", nargs=2, type=float, default=[-5.0, 5.0],
                       help="Y axis bounds in meters (default: -5 5)")
    parser.add_argument("--z-bounds", nargs=2, type=float, default=[0.0, 10.0],
                       help="Z axis bounds in meters (default: 0 10)")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        return 1
    
    # Get video dimensions
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        return 1
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Check if side-by-side format
    if width > height * 1.5:
        # Side-by-side format, use half width
        width = width // 2
    
    cap.release()
    
    # Create camera intrinsics
    camera_intrinsics = CameraIntrinsics.from_fov(width, height, args.fov)
    
    # Create voxel configuration
    voxel_config = VoxelGridConfig(
        voxel_size=args.voxel_size,
        depth_scale=args.depth_scale,
        world_scale=args.world_scale,
        min_depth=0.1,
        max_depth=20.0,
        x_min=args.x_bounds[0],
        x_max=args.x_bounds[1],
        y_min=args.y_bounds[0],
        y_max=args.y_bounds[1],
        z_min=args.z_bounds[0],
        z_max=args.z_bounds[1],
        occupancy_threshold=0.3,
        confidence_weight=True,
        outlier_removal=True
    )
    
    # Create processor and run
    processor = VideoToVoxelProcessor(camera_intrinsics, voxel_config)
    
    try:
        processor.process_video(
            args.video,
            args.output,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 