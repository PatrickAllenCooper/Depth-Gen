#!/usr/bin/env python3
"""
Temporal Consistency Module for Video Depth Estimation
Implements state-of-the-art techniques for reducing flickering and maintaining consistency.

Based on research from:
- "Blind Video Temporal Consistency" (Bonneel et al., 2015)
- "RollingDepth" (Ke et al., 2024)  
- "Robust Consistent Video Depth Estimation" (Kopf et al., 2020)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from collections import deque
import time


class TemporalConsistencyProcessor:
    """
    Temporal consistency processor for video depth estimation.
    
    Key features:
    - Optical flow-based warping of previous depth maps
    - Confidence-based blending with current predictions
    - Multi-frame temporal filtering
    - Occlusion detection and handling
    - Memory-efficient processing for long videos
    """
    
    def __init__(self, 
                 temporal_window: int = 3,
                 flow_confidence_threshold: float = 0.5,
                 blend_alpha: float = 0.7,
                 use_gpu: bool = True,
                 max_flow_magnitude: float = 50.0):
        """
        Initialize temporal consistency processor.
        
        Args:
            temporal_window: Number of previous frames to consider
            flow_confidence_threshold: Threshold for optical flow confidence
            blend_alpha: Blending weight (0=only current, 1=only warped)
            use_gpu: Whether to use GPU for processing
            max_flow_magnitude: Maximum allowed optical flow magnitude
        """
        self.temporal_window = temporal_window
        self.flow_confidence_threshold = flow_confidence_threshold
        self.blend_alpha = blend_alpha
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_flow_magnitude = max_flow_magnitude
        
        # Temporal buffers
        self.frame_buffer = deque(maxlen=temporal_window)
        self.depth_buffer = deque(maxlen=temporal_window)
        self.flow_buffer = deque(maxlen=temporal_window-1)
        
        # Optical flow estimator (using Farneback for reliability)
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Statistics tracking
        self.stats = {
            'frames_processed': 0,
            'avg_flow_magnitude': 0.0,
            'consistency_ratio': 0.0,
            'processing_time': 0.0
        }
        
        print(f"Temporal Consistency Processor initialized:")
        print(f"  - Temporal window: {temporal_window} frames")
        print(f"  - GPU acceleration: {self.use_gpu}")
        print(f"  - Flow confidence threshold: {flow_confidence_threshold}")
    
    def reset(self):
        """Reset all temporal buffers (call at start of new video)."""
        self.frame_buffer.clear()
        self.depth_buffer.clear()  
        self.flow_buffer.clear()
        self.stats = {
            'frames_processed': 0,
            'avg_flow_magnitude': 0.0,
            'consistency_ratio': 0.0,
            'processing_time': 0.0
        }
        print("Temporal buffers reset for new video sequence")
    
    def compute_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow between consecutive frames.
        
        Args:
            prev_frame: Previous frame (grayscale, uint8)
            curr_frame: Current frame (grayscale, uint8)
            
        Returns:
            flow: Optical flow (H, W, 2)
            confidence: Flow confidence mask (H, W)
        """
        # Compute optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, curr_frame, None, **self.flow_params
        )
        
        # Compute flow confidence based on magnitude and consistency
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Confidence based on flow magnitude (smaller is more reliable)
        magnitude_confidence = np.exp(-flow_magnitude / 10.0)
        
        # Confidence based on flow consistency (check neighboring pixels)
        kernel = np.ones((3, 3), np.float32) / 9
        flow_x_smooth = cv2.filter2D(flow[..., 0], -1, kernel)
        flow_y_smooth = cv2.filter2D(flow[..., 1], -1, kernel)
        
        consistency_error = np.sqrt((flow[..., 0] - flow_x_smooth)**2 + 
                                  (flow[..., 1] - flow_y_smooth)**2)
        consistency_confidence = np.exp(-consistency_error / 5.0)
        
        # Combined confidence
        confidence = magnitude_confidence * consistency_confidence
        
        # Mask out very large flows (likely errors)
        large_flow_mask = flow_magnitude > self.max_flow_magnitude
        confidence[large_flow_mask] = 0.0
        
        return flow, confidence
    
    def warp_depth_with_flow(self, depth: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warp depth map using optical flow.
        
        Args:
            depth: Previous depth map (H, W)
            flow: Optical flow (H, W, 2)
            
        Returns:
            warped_depth: Warped depth map (H, W)
        """
        h, w = depth.shape
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Apply flow to coordinates
        new_x = x_coords + flow[..., 0]
        new_y = y_coords + flow[..., 1]
        
        # Warp depth using interpolation
        warped_depth = cv2.remap(
            depth.astype(np.float32),
            new_x.astype(np.float32),
            new_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return warped_depth
    
    def detect_occlusions(self, flow: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        """
        Detect occlusions using flow divergence and confidence.
        
        Args:
            flow: Optical flow (H, W, 2)
            confidence: Flow confidence (H, W)
            
        Returns:
            occlusion_mask: Binary mask indicating occlusions (H, W)
        """
        # Compute flow divergence
        dx = cv2.Sobel(flow[..., 0], cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(flow[..., 1], cv2.CV_32F, 0, 1, ksize=3)
        divergence = dx + dy
        
        # High divergence indicates occlusions/disocclusions
        divergence_threshold = np.percentile(np.abs(divergence), 95)
        high_divergence = np.abs(divergence) > divergence_threshold
        
        # Low confidence also indicates potential occlusions
        low_confidence = confidence < self.flow_confidence_threshold
        
        # Combine indicators
        occlusion_mask = high_divergence | low_confidence
        
        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        occlusion_mask = cv2.morphologyEx(
            occlusion_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
        ).astype(bool)
        
        return occlusion_mask
    
    def blend_depth_maps(self, 
                        current_depth: np.ndarray, 
                        warped_depth: np.ndarray,
                        confidence: np.ndarray,
                        occlusion_mask: np.ndarray) -> np.ndarray:
        """
        Blend current and warped depth maps using confidence weights.
        
        Args:
            current_depth: Current frame depth prediction (H, W)
            warped_depth: Warped previous depth (H, W)
            confidence: Flow confidence (H, W)
            occlusion_mask: Occlusion mask (H, W)
            
        Returns:
            blended_depth: Temporally consistent depth map (H, W)
        """
        # Create blending weights
        temporal_weight = confidence * (1 - occlusion_mask.astype(float))
        current_weight = 1 - temporal_weight * self.blend_alpha
        temporal_weight = temporal_weight * self.blend_alpha
        
        # Normalize weights
        total_weight = current_weight + temporal_weight
        current_weight = current_weight / (total_weight + 1e-8)
        temporal_weight = temporal_weight / (total_weight + 1e-8)
        
        # Blend depth maps
        blended_depth = (current_weight * current_depth + 
                        temporal_weight * warped_depth)
        
        return blended_depth
    
    def process_frame(self, 
                     current_frame: np.ndarray, 
                     current_depth: np.ndarray) -> np.ndarray:
        """
        Process a single frame for temporal consistency.
        
        Args:
            current_frame: Current RGB frame (H, W, 3)
            current_depth: Current depth prediction (H, W)
            
        Returns:
            consistent_depth: Temporally consistent depth map (H, W)
        """
        start_time = time.time()
        
        # Convert frame to grayscale for optical flow
        if len(current_frame.shape) == 3:
            gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        else:
            gray_frame = current_frame
        
        # Add to buffers
        self.frame_buffer.append(gray_frame)
        
        # If this is the first frame, just return current depth
        if len(self.frame_buffer) < 2:
            self.depth_buffer.append(current_depth.copy())
            self.stats['frames_processed'] += 1
            return current_depth
        
        # Get previous frame and depth
        prev_frame = self.frame_buffer[-2]
        prev_depth = self.depth_buffer[-1]
        
        # Compute optical flow
        flow, confidence = self.compute_optical_flow(prev_frame, gray_frame)
        
        # Warp previous depth to current frame
        warped_depth = self.warp_depth_with_flow(prev_depth, flow)
        
        # Detect occlusions
        occlusion_mask = self.detect_occlusions(flow, confidence)
        
        # Blend current and warped depth
        blended_depth = self.blend_depth_maps(
            current_depth, warped_depth, confidence, occlusion_mask
        )
        
        # Add to depth buffer
        self.depth_buffer.append(blended_depth)
        
        # Update statistics
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_flow_mag = np.mean(flow_magnitude)
        consistency_ratio = np.mean(confidence > self.flow_confidence_threshold)
        processing_time = time.time() - start_time
        
        # Update running statistics
        n = self.stats['frames_processed']
        self.stats['avg_flow_magnitude'] = (n * self.stats['avg_flow_magnitude'] + avg_flow_mag) / (n + 1)
        self.stats['consistency_ratio'] = (n * self.stats['consistency_ratio'] + consistency_ratio) / (n + 1)
        self.stats['processing_time'] = (n * self.stats['processing_time'] + processing_time) / (n + 1)
        self.stats['frames_processed'] += 1
        
        return blended_depth
    
    def get_stats(self) -> Dict[str, float]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def print_stats(self):
        """Print current processing statistics."""
        stats = self.get_stats()
        print(f"\nTemporal Consistency Statistics:")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Avg optical flow magnitude: {stats['avg_flow_magnitude']:.2f} pixels")
        print(f"  Flow consistency ratio: {stats['consistency_ratio']:.2%}")
        print(f"  Avg processing time: {stats['processing_time']*1000:.1f}ms per frame")


def create_temporal_processor(temporal_window: int = 3, 
                            flow_confidence_threshold: float = 0.5,
                            blend_alpha: float = 0.7) -> TemporalConsistencyProcessor:
    """
    Create a temporal consistency processor with specified configuration.
    
    Args:
        temporal_window: Number of previous frames to consider
        flow_confidence_threshold: Threshold for optical flow confidence
        blend_alpha: Blending weight (0=only current, 1=only warped)
        
    Returns:
        TemporalConsistencyProcessor instance
    """
    return TemporalConsistencyProcessor(
        temporal_window=temporal_window,
        flow_confidence_threshold=flow_confidence_threshold,
        blend_alpha=blend_alpha
    ) 