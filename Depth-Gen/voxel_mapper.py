#!/usr/bin/env python3
"""
Voxel Mapping Module for Depth Frames
Converts depth maps to 3D voxel representations with physical realism.

This module implements:
- Depth to point cloud conversion with camera intrinsics
- Voxel grid generation and occupancy mapping
- Multi-frame integration with confidence weighting
- Physical scaling and coordinate systems
- Export to various 3D formats
"""

import numpy as np
import cv2
import torch
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
import open3d as o3d
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from scipy.ndimage import binary_dilation


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for depth to 3D conversion."""
    fx: float  # Focal length in x
    fy: float  # Focal length in y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int
    
    @classmethod
    def from_fov(cls, width: int, height: int, fov_degrees: float = 60.0):
        """Create intrinsics from field of view."""
        fov_rad = np.deg2rad(fov_degrees)
        fx = fy = width / (2 * np.tan(fov_rad / 2))
        cx = width / 2
        cy = height / 2
        return cls(fx, fy, cx, cy, width, height)
    
    @property
    def matrix(self) -> np.ndarray:
        """Get 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])


@dataclass
class VoxelGridConfig:
    """Configuration for voxel grid generation."""
    voxel_size: float = 0.05  # 5cm voxels for good detail
    min_depth: float = 0.1    # Minimum depth in meters
    max_depth: float = 10.0   # Maximum depth in meters
    occupancy_threshold: float = 0.5  # Threshold for occupied voxels
    confidence_weight: bool = True  # Use confidence weighting
    outlier_removal: bool = True  # Remove statistical outliers
    
    # Physical scale parameters
    depth_scale: float = 1000.0  # Depth units to meters (1000 = mm to m)
    world_scale: float = 1.0     # Additional world scaling
    
    # Voxel grid bounds (in meters)
    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -5.0
    y_max: float = 5.0
    z_min: float = 0.0
    z_max: float = 10.0


class VoxelMapper:
    """
    Main voxel mapping class for converting depth frames to 3D voxel representations.
    """
    
    def __init__(self, 
                 camera_intrinsics: CameraIntrinsics,
                 voxel_config: VoxelGridConfig = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize voxel mapper.
        
        Args:
            camera_intrinsics: Camera parameters for 3D reconstruction
            voxel_config: Voxel grid configuration
            device: Device for computation ('cuda' or 'cpu')
        """
        self.camera_intrinsics = camera_intrinsics
        self.voxel_config = voxel_config or VoxelGridConfig()
        self.device = device
        
        # Initialize voxel grid
        self._init_voxel_grid()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'points_generated': 0,
            'voxels_occupied': 0,
            'outliers_removed': 0
        }
        
        print(f"Voxel Mapper initialized:")
        print(f"  - Camera: {camera_intrinsics.width}x{camera_intrinsics.height}")
        print(f"  - Voxel size: {self.voxel_config.voxel_size}m")
        print(f"  - Grid bounds: X[{self.voxel_config.x_min}, {self.voxel_config.x_max}]m")
        print(f"  - Device: {device}")
    
    def _init_voxel_grid(self):
        """Initialize voxel grid data structures."""
        # Calculate grid dimensions
        self.grid_size_x = int((self.voxel_config.x_max - self.voxel_config.x_min) / self.voxel_config.voxel_size)
        self.grid_size_y = int((self.voxel_config.y_max - self.voxel_config.y_min) / self.voxel_config.voxel_size)
        self.grid_size_z = int((self.voxel_config.z_max - self.voxel_config.z_min) / self.voxel_config.voxel_size)
        
        # Voxel occupancy grid (probability of occupation)
        self.voxel_grid = np.zeros((self.grid_size_x, self.grid_size_y, self.grid_size_z), dtype=np.float32)
        
        # Confidence accumulator for weighted averaging
        self.confidence_grid = np.zeros_like(self.voxel_grid)
        
        # Color information for each voxel (optional)
        self.color_grid = np.zeros((self.grid_size_x, self.grid_size_y, self.grid_size_z, 3), dtype=np.float32)
        
        print(f"  - Grid size: {self.grid_size_x}x{self.grid_size_y}x{self.grid_size_z} voxels")
    
    def depth_to_point_cloud(self, 
                           depth_map: np.ndarray, 
                           color_image: Optional[np.ndarray] = None,
                           depth_confidence: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            depth_map: Depth map (H, W) in depth units
            color_image: Optional RGB image (H, W, 3)
            depth_confidence: Optional confidence map (H, W)
            
        Returns:
            points: 3D points (N, 3) in meters
            colors: Optional colors (N, 3)
            confidences: Optional confidence values (N,)
        """
        h, w = depth_map.shape
        
        # Create pixel coordinate grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert depth to meters
        depth_meters = depth_map.astype(np.float32) / self.voxel_config.depth_scale
        
        # Filter by depth range
        valid_mask = (depth_meters > self.voxel_config.min_depth) & \
                    (depth_meters < self.voxel_config.max_depth) & \
                    (depth_meters > 0)
        
        # Back-project to 3D
        xx_valid = xx[valid_mask]
        yy_valid = yy[valid_mask]
        depth_valid = depth_meters[valid_mask]
        
        # Apply inverse camera intrinsics
        x_3d = (xx_valid - self.camera_intrinsics.cx) * depth_valid / self.camera_intrinsics.fx
        y_3d = (yy_valid - self.camera_intrinsics.cy) * depth_valid / self.camera_intrinsics.fy
        z_3d = depth_valid
        
        # Stack into points
        points = np.stack([x_3d, y_3d, z_3d], axis=-1)
        
        # Apply world scaling
        points *= self.voxel_config.world_scale
        
        # Extract colors if provided
        colors = None
        if color_image is not None:
            colors = color_image[valid_mask].astype(np.float32) / 255.0
        
        # Extract confidences if provided
        confidences = None
        if depth_confidence is not None:
            confidences = depth_confidence[valid_mask]
        
        self.stats['points_generated'] += len(points)
        
        return points, colors, confidences
    
    def remove_outliers(self, points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        """
        Remove statistical outliers from point cloud.
        
        Args:
            points: 3D points (N, 3)
            nb_neighbors: Number of neighbors for outlier detection
            std_ratio: Standard deviation multiplier for outlier threshold
            
        Returns:
            inlier_indices: Indices of inlier points
        """
        if len(points) < nb_neighbors:
            return np.arange(len(points))
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Remove outliers
        _, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        
        outliers_removed = len(points) - len(inlier_indices)
        self.stats['outliers_removed'] += outliers_removed
        
        return np.array(inlier_indices)
    
    def points_to_voxel_indices(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 3D points to voxel grid indices.
        
        Args:
            points: 3D points (N, 3) in meters
            
        Returns:
            voxel_indices: Valid voxel indices (M, 3)
            point_indices: Corresponding point indices (M,)
        """
        # Convert to voxel coordinates
        voxel_coords = (points - np.array([self.voxel_config.x_min, 
                                          self.voxel_config.y_min, 
                                          self.voxel_config.z_min])) / self.voxel_config.voxel_size
        
        # Round to nearest voxel
        voxel_indices = np.round(voxel_coords).astype(int)
        
        # Filter valid indices
        valid_mask = (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < self.grid_size_x) & \
                    (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < self.grid_size_y) & \
                    (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < self.grid_size_z)
        
        valid_voxel_indices = voxel_indices[valid_mask]
        valid_point_indices = np.where(valid_mask)[0]
        
        return valid_voxel_indices, valid_point_indices
    
    def update_voxel_grid(self, 
                         points: np.ndarray, 
                         colors: Optional[np.ndarray] = None,
                         confidences: Optional[np.ndarray] = None):
        """
        Update voxel grid with new points.
        
        Args:
            points: 3D points (N, 3)
            colors: Optional colors (N, 3)
            confidences: Optional confidence values (N,)
        """
        # Remove outliers if enabled
        if self.voxel_config.outlier_removal and len(points) > 100:
            inlier_indices = self.remove_outliers(points)
            points = points[inlier_indices]
            if colors is not None:
                colors = colors[inlier_indices]
            if confidences is not None:
                confidences = confidences[inlier_indices]
        
        # Convert to voxel indices
        voxel_indices, point_indices = self.points_to_voxel_indices(points)
        
        if len(voxel_indices) == 0:
            return
        
        # Default confidence if not provided
        if confidences is None:
            confidences = np.ones(len(points))
        
        # Update occupancy with confidence weighting
        for i, (vx, vy, vz) in enumerate(voxel_indices):
            point_idx = point_indices[i]
            confidence = confidences[point_idx]
            
            if self.voxel_config.confidence_weight:
                # Weighted update
                old_confidence = self.confidence_grid[vx, vy, vz]
                new_confidence = old_confidence + confidence
                
                # Update occupancy as weighted average
                self.voxel_grid[vx, vy, vz] = (
                    (self.voxel_grid[vx, vy, vz] * old_confidence + confidence) / new_confidence
                )
                self.confidence_grid[vx, vy, vz] = new_confidence
            else:
                # Simple maximum
                self.voxel_grid[vx, vy, vz] = max(self.voxel_grid[vx, vy, vz], confidence)
            
            # Update color if provided
            if colors is not None:
                color = colors[point_idx]
                if self.voxel_config.confidence_weight:
                    # Weighted color update
                    old_weight = self.confidence_grid[vx, vy, vz] - confidence
                    self.color_grid[vx, vy, vz] = (
                        (self.color_grid[vx, vy, vz] * old_weight + color * confidence) / 
                        self.confidence_grid[vx, vy, vz]
                    )
                else:
                    self.color_grid[vx, vy, vz] = color
    
    def process_depth_frame(self, 
                          depth_path: str, 
                          color_path: Optional[str] = None,
                          pose: Optional[np.ndarray] = None) -> Dict[str, int]:
        """
        Process a single depth frame and update voxel grid.
        
        Args:
            depth_path: Path to depth map image
            color_path: Optional path to color image
            pose: Optional 4x4 camera pose matrix (identity if not provided)
            
        Returns:
            stats: Processing statistics
        """
        # Load depth map
        depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_map is None:
            depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        
        # Load color image if provided
        color_image = None
        if color_path and Path(color_path).exists():
            color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # Generate confidence map (simple gradient-based)
        depth_confidence = self._compute_depth_confidence(depth_map)
        
        # Convert to point cloud
        points, colors, confidences = self.depth_to_point_cloud(
            depth_map, color_image, depth_confidence
        )
        
        # Apply pose transformation if provided
        if pose is not None:
            points_homo = np.hstack([points, np.ones((len(points), 1))])
            points_transformed = (pose @ points_homo.T).T[:, :3]
            points = points_transformed
        
        # Update voxel grid
        self.update_voxel_grid(points, colors, confidences)
        
        self.stats['frames_processed'] += 1
        
        return {
            'points_generated': len(points),
            'voxels_updated': len(np.unique(self.points_to_voxel_indices(points)[0], axis=0))
        }
    
    def _compute_depth_confidence(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Compute confidence map from depth gradients.
        
        Args:
            depth_map: Input depth map
            
        Returns:
            confidence: Confidence map (0-1)
        """
        # Compute gradients
        grad_x = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and invert (low gradient = high confidence)
        grad_mag_norm = grad_mag / (np.percentile(grad_mag, 95) + 1e-6)
        confidence = np.exp(-grad_mag_norm)
        
        # Reduce confidence at depth discontinuities
        depth_range = np.percentile(depth_map[depth_map > 0], 99) - np.percentile(depth_map[depth_map > 0], 1)
        large_grad_threshold = depth_range * 0.1
        confidence[grad_mag > large_grad_threshold] *= 0.5
        
        return confidence
    
    def get_occupied_voxels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get occupied voxel centers and properties.
        
        Returns:
            centers: Voxel centers (N, 3) in meters
            occupancies: Occupancy values (N,)
            colors: Voxel colors (N, 3)
        """
        # Get occupied voxels above threshold
        occupied_mask = self.voxel_grid > self.voxel_config.occupancy_threshold
        occupied_indices = np.argwhere(occupied_mask)
        
        if len(occupied_indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert to world coordinates
        centers = occupied_indices * self.voxel_config.voxel_size + \
                 np.array([self.voxel_config.x_min, self.voxel_config.y_min, self.voxel_config.z_min])
        
        # Get occupancy values
        occupancies = self.voxel_grid[occupied_mask]
        
        # Get colors
        colors = self.color_grid[occupied_mask]
        
        self.stats['voxels_occupied'] = len(centers)
        
        return centers, occupancies, colors
    
    def export_to_mesh(self, output_path: str, use_marching_cubes: bool = True):
        """
        Export voxel grid as 3D mesh.
        
        Args:
            output_path: Output file path (.ply, .obj, .stl)
            use_marching_cubes: Use marching cubes for smooth surface
        """
        if use_marching_cubes:
            # Smooth surface using marching cubes
            from skimage import measure
            
            # Pad grid to ensure closed surface
            padded_grid = np.pad(self.voxel_grid, 1, mode='constant', constant_values=0)
            
            # Extract surface
            verts, faces, normals, _ = measure.marching_cubes(
                padded_grid, 
                level=self.voxel_config.occupancy_threshold,
                spacing=(self.voxel_config.voxel_size,) * 3
            )
            
            # Adjust for padding and transform to world coordinates
            verts = verts - 1  # Remove padding offset
            verts = verts * self.voxel_config.voxel_size + \
                   np.array([self.voxel_config.x_min, self.voxel_config.y_min, self.voxel_config.z_min])
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            
            # Simplify if too complex
            if len(mesh.faces) > 100000:
                # Calculate target reduction ratio
                target_faces = 50000
                reduction_ratio = 1.0 - (target_faces / len(mesh.faces))
                mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            
        else:
            # Voxel-based mesh
            centers, occupancies, colors = self.get_occupied_voxels()
            
            if len(centers) == 0:
                print("No occupied voxels to export!")
                return
            
            # Create boxes for each voxel
            mesh = trimesh.voxel.ops.multibox(
                centers=centers,
                pitch=self.voxel_config.voxel_size,
                colors=colors if np.any(colors > 0) else None
            )
        
        # Export mesh
        mesh.export(output_path)
        print(f"Exported mesh to: {output_path}")
        print(f"  - Vertices: {len(mesh.vertices)}")
        print(f"  - Faces: {len(mesh.faces)}")
    
    def export_to_pointcloud(self, output_path: str):
        """
        Export voxel centers as point cloud.
        
        Args:
            output_path: Output file path (.ply, .xyz, .pcd)
        """
        centers, occupancies, colors = self.get_occupied_voxels()
        
        if len(centers) == 0:
            print("No occupied voxels to export!")
            return
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers)
        
        if np.any(colors > 0):
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save point cloud
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Exported point cloud to: {output_path}")
        print(f"  - Points: {len(centers)}")
    
    def visualize_voxels(self, save_path: Optional[str] = None, show: bool = True):
        """
        Visualize voxel grid using matplotlib.
        
        Args:
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        centers, occupancies, colors = self.get_occupied_voxels()
        
        if len(centers) == 0:
            print("No occupied voxels to visualize!")
            return
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample voxels if too many
        if len(centers) > 10000:
            indices = np.random.choice(len(centers), 10000, replace=False)
            centers = centers[indices]
            occupancies = occupancies[indices]
            colors = colors[indices]
        
        # Plot voxels
        if np.any(colors > 0):
            scatter = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                               c=colors, s=20, alpha=0.6, edgecolor='none')
        else:
            scatter = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                               c=occupancies, cmap='viridis', s=20, alpha=0.6, edgecolor='none')
            plt.colorbar(scatter, ax=ax, label='Occupancy')
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)') 
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Voxel Map ({len(centers)} occupied voxels)')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set view angle
        ax.view_init(elev=20, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_voxel_grid(self, output_path: str):
        """
        Save voxel grid data to file.
        
        Args:
            output_path: Output file path (.npz)
        """
        np.savez_compressed(
            output_path,
            voxel_grid=self.voxel_grid,
            confidence_grid=self.confidence_grid,
            color_grid=self.color_grid,
            config=self.voxel_config.__dict__,
            camera_intrinsics=self.camera_intrinsics.__dict__,
            stats=self.stats
        )
        print(f"Saved voxel grid to: {output_path}")
    
    def load_voxel_grid(self, input_path: str):
        """
        Load voxel grid data from file.
        
        Args:
            input_path: Input file path (.npz)
        """
        data = np.load(input_path)
        self.voxel_grid = data['voxel_grid']
        self.confidence_grid = data['confidence_grid']
        self.color_grid = data['color_grid']
        self.stats = dict(data['stats'].item())
        print(f"Loaded voxel grid from: {input_path}")
    
    def print_stats(self):
        """Print processing statistics."""
        print("\nVoxel Mapping Statistics:")
        print(f"  - Frames processed: {self.stats['frames_processed']}")
        print(f"  - Total points generated: {self.stats['points_generated']:,}")
        print(f"  - Occupied voxels: {self.stats['voxels_occupied']:,}")
        print(f"  - Outliers removed: {self.stats['outliers_removed']:,}")
        print(f"  - Voxel occupancy rate: {self.stats['voxels_occupied'] / (self.grid_size_x * self.grid_size_y * self.grid_size_z):.2%}")


def process_test_frames(frame_dir: str = "test_frames", 
                       output_dir: str = "voxel_output",
                       camera_fov: float = 60.0):
    """
    Process test depth frames and generate voxel map.
    
    Args:
        frame_dir: Directory containing test frames
        output_dir: Output directory for voxel data
        camera_fov: Camera field of view in degrees
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find depth frames
    frame_path = Path(frame_dir)
    depth_files = sorted(frame_path.glob("*_depth.png"))
    
    if not depth_files:
        print(f"No depth frames found in {frame_dir}")
        return
    
    print(f"Found {len(depth_files)} depth frames")
    
    # Get image dimensions from first frame
    first_depth = cv2.imread(str(depth_files[0]), cv2.IMREAD_ANYDEPTH)
    if first_depth is None:
        first_depth = cv2.imread(str(depth_files[0]), cv2.IMREAD_GRAYSCALE)
    
    height, width = first_depth.shape
    
    # Create camera intrinsics
    camera_intrinsics = CameraIntrinsics.from_fov(width, height, camera_fov)
    
    # Create voxel mapper
    voxel_config = VoxelGridConfig(
        voxel_size=0.02,  # 2cm voxels for detail
        depth_scale=255.0,  # Assuming 8-bit depth maps normalized to 0-255
        world_scale=0.01,   # Scale to realistic room size
        x_min=-3.0, x_max=3.0,
        y_min=-3.0, y_max=3.0,
        z_min=0.0, z_max=6.0
    )
    
    mapper = VoxelMapper(camera_intrinsics, voxel_config)
    
    # Process each depth frame
    print("\nProcessing depth frames...")
    for depth_file in tqdm(depth_files):
        # Find corresponding color image
        color_file = str(depth_file).replace("_depth.png", ".png")
        if not Path(color_file).exists():
            color_file = None
        
        # Process frame
        stats = mapper.process_depth_frame(str(depth_file), color_file)
        
    # Print statistics
    mapper.print_stats()
    
    # Export results
    print("\nExporting results...")
    
    # Save voxel grid data
    mapper.save_voxel_grid(str(output_path / "voxel_grid.npz"))
    
    # Export as mesh
    mapper.export_to_mesh(str(output_path / "voxel_mesh.ply"), use_marching_cubes=True)
    mapper.export_to_mesh(str(output_path / "voxel_cubes.ply"), use_marching_cubes=False)
    
    # Export as point cloud
    mapper.export_to_pointcloud(str(output_path / "voxel_points.ply"))
    
    # Generate visualization
    mapper.visualize_voxels(str(output_path / "voxel_visualization.png"), show=False)
    
    print(f"\n✓ Voxel mapping complete! Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert depth frames to voxel map")
    parser.add_argument("--input-dir", default="test_frames", help="Input directory with depth frames")
    parser.add_argument("--output-dir", default="voxel_output", help="Output directory for voxel data")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera field of view in degrees")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size in meters")
    
    args = parser.parse_args()
    
    # Process test frames
    process_test_frames(args.input_dir, args.output_dir, args.fov) 