#!/usr/bin/env python3
"""
Test script for temporal consistency in video depth estimation.
Compares depth maps with and without temporal consistency.
"""

import cv2
import numpy as np
import torch
import os
from PIL import Image
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
from temporal_consistency import create_temporal_processor
import time


def load_depth_model():
    """Load the DepthPro model for testing."""
    print("Loading DepthPro model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_repo = "apple/DepthPro-hf"
    image_processor = DepthProImageProcessorFast.from_pretrained(model_repo)
    model = DepthProForDepthEstimation.from_pretrained(
        model_repo,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return image_processor, model, device


def predict_depth_raw(image_processor, model, device, img):
    """Predict depth without temporal consistency."""
    inputs = image_processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
    
    post_processed = image_processor.post_process_depth_estimation(
        outputs, target_sizes=[(img.height, img.width)]
    )[0]
    
    depth = post_processed["predicted_depth"]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.clip(0, 255).to(torch.uint8)
    depth_np = depth.detach().cpu().numpy()
    
    return depth_np


def extract_test_frames(video_path, num_frames=10):
    """Extract a sequence of test frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract frames from middle of video (more likely to have motion)
    start_frame = total_frames // 3
    frames = []
    
    for i in range(num_frames):
        frame_idx = start_frame + i * 2  # Every 2nd frame for more motion
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            break
    
    cap.release()
    print(f"Extracted {len(frames)} test frames")
    return frames


def test_temporal_consistency():
    """Test temporal consistency on a sequence of frames."""
    video_path = "test_depth_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Test video not found: {video_path}")
        print("Please ensure test_depth_video.mp4 is in the current directory")
        return
    
    # Load model
    image_processor, model, device = load_depth_model()
    
    # Create temporal consistency processor
    temporal_processor = create_temporal_processor(
        temporal_window=3,
        flow_confidence_threshold=0.6,
        blend_alpha=0.75
    )
    
    # Extract test frames
    frames = extract_test_frames(video_path, num_frames=8)
    
    if len(frames) < 3:
        print("Not enough frames extracted for testing")
        return
    
    print(f"\nProcessing {len(frames)} frames...")
    
    # Process frames with and without temporal consistency
    raw_depths = []
    consistent_depths = []
    processing_times = []
    
    # Reset temporal processor
    temporal_processor.reset()
    
    for i, frame_rgb in enumerate(frames):
        print(f"Processing frame {i+1}/{len(frames)}")
        
        # Convert to PIL
        pil_img = Image.fromarray(frame_rgb)
        
        start_time = time.time()
        
        # Raw depth prediction
        raw_depth = predict_depth_raw(image_processor, model, device, pil_img)
        
        # Temporal consistency processing
        consistent_depth = temporal_processor.process_frame(frame_rgb, raw_depth.copy())
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        raw_depths.append(raw_depth)
        consistent_depths.append(consistent_depth)
        
        # Clear GPU cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Analyze results
    print(f"\n{'='*60}")
    print("TEMPORAL CONSISTENCY TEST RESULTS")
    print(f"{'='*60}")
    
    # Print temporal processor statistics
    temporal_processor.print_stats()
    
    # Calculate temporal differences (measure of flickering)
    raw_diffs = []
    consistent_diffs = []
    
    for i in range(1, len(frames)):
        raw_diff = np.mean(np.abs(raw_depths[i].astype(float) - raw_depths[i-1].astype(float)))
        consistent_diff = np.mean(np.abs(consistent_depths[i].astype(float) - consistent_depths[i-1].astype(float)))
        
        raw_diffs.append(raw_diff)
        consistent_diffs.append(consistent_diff)
    
    avg_raw_diff = np.mean(raw_diffs)
    avg_consistent_diff = np.mean(consistent_diffs)
    
    print(f"\nTemporal Stability Analysis:")
    print(f"  Raw depth frame differences: {avg_raw_diff:.2f} (higher = more flickering)")
    print(f"  Consistent depth differences: {avg_consistent_diff:.2f}")
    print(f"  Improvement: {((avg_raw_diff - avg_consistent_diff) / avg_raw_diff * 100):.1f}% reduction in flickering")
    
    # Performance analysis
    avg_processing_time = np.mean(processing_times)
    print(f"\nPerformance:")
    print(f"  Average processing time: {avg_processing_time*1000:.1f}ms per frame")
    print(f"  Temporal consistency overhead: ~{temporal_processor.get_stats()['processing_time']*1000:.1f}ms per frame")
    
    # Save sample comparison
    os.makedirs("temporal_test_results", exist_ok=True)
    
    # Save first, middle, and last frames for comparison
    test_indices = [0, len(frames)//2, len(frames)-1]
    
    for idx in test_indices:
        # Original frame
        Image.fromarray(frames[idx]).save(f"temporal_test_results/frame_{idx:02d}_original.png")
        
        # Raw depth
        Image.fromarray(raw_depths[idx].astype(np.uint8)).save(f"temporal_test_results/frame_{idx:02d}_raw_depth.png")
        
        # Consistent depth
        Image.fromarray(consistent_depths[idx].astype(np.uint8)).save(f"temporal_test_results/frame_{idx:02d}_consistent_depth.png")
        
        # Side-by-side comparison
        comparison = np.hstack([
            frames[idx].astype(np.uint8),
            np.stack([raw_depths[idx].astype(np.uint8)]*3, axis=2),
            np.stack([consistent_depths[idx].astype(np.uint8)]*3, axis=2)
        ])
        Image.fromarray(comparison.astype(np.uint8)).save(f"temporal_test_results/frame_{idx:02d}_comparison.png")
    
    print(f"\n✓ Sample results saved to 'temporal_test_results/' directory")
    print(f"✓ Check the comparison images to see the difference")
    
    # Recommendation
    improvement_pct = (avg_raw_diff - avg_consistent_diff) / avg_raw_diff * 100
    if improvement_pct > 20:
        print(f"\n🎉 Temporal consistency shows significant improvement ({improvement_pct:.1f}% flicker reduction)")
        print("   Recommended for production use!")
    elif improvement_pct > 10:
        print(f"\n✅ Temporal consistency shows moderate improvement ({improvement_pct:.1f}% flicker reduction)")
        print("   Beneficial for most videos")
    else:
        print(f"\n⚠️  Temporal consistency shows minimal improvement ({improvement_pct:.1f}% flicker reduction)")
        print("   May not be necessary for this video type")


if __name__ == "__main__":
    print("Temporal Consistency Test for Video Depth Estimation")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Using CPU (slower)")
    
    print()
    test_temporal_consistency() 