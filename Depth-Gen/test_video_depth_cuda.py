#!/usr/bin/env python3
"""
CUDA-optimized test script for Windows with RTX 3080.
Extracts frames from video and generates depth maps with CUDA optimizations.
"""

import os
import cv2
import torch
import gc
from PIL import Image
import numpy as np
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
from pathlib import Path
import time

# Device selection optimized for CUDA
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# Model loading with CUDA optimizations
MODEL_REPO = os.getenv("DEPTH_MODEL_REPO", "apple/DepthPro-hf")
print(f"Loading model from: {MODEL_REPO}")

def load_model_optimized():
    """Load model with CUDA optimizations for RTX 3080"""
    try:
        print("Loading image processor...")
        image_processor = DepthProImageProcessorFast.from_pretrained(MODEL_REPO)
        
        print("Loading depth estimation model...")
        model = DepthProForDepthEstimation.from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.float16,  # Use half precision for RTX 3080
            low_cpu_mem_usage=True
        ).to(DEVICE)
        
        model.eval()
        
        # CUDA-specific optimizations
        if DEVICE.type == "cuda":
            print("Applying CUDA optimizations...")
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            # Enable TensorFloat-32 for RTX 30 series
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Compile model for better performance (if available)
            if hasattr(torch, 'compile'):
                print("Compiling model for CUDA...")
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    print("Model compilation successful!")
                except Exception as e:
                    print(f"Model compilation failed (continuing without): {e}")
        
        print("Model loaded and optimized successfully!")
        return image_processor, model
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

image_processor, model = load_model_optimized()

def predict_depth_optimized(img: Image.Image, max_size: int = 1536) -> Image.Image:
    """Generate depth map with CUDA optimizations for RTX 3080."""
    start_time = time.time()
    
    # RTX 3080 can handle larger images - increase max_size
    original_size = img.size
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"  Resized from {original_size} to {img.size} for processing")
    
    # Clear CUDA cache before processing
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    
    try:
        # Use mixed precision for better performance on RTX 3080
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
            inputs = image_processor(images=img, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            post_processed = image_processor.post_process_depth_estimation(
                outputs, target_sizes=[(img.height, img.width)]
            )[0]
            
            depth = post_processed["predicted_depth"]
            
            # Normalize to 0-255 range
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.clip(0, 255).to(torch.uint8)
            depth_np = depth.detach().cpu().numpy()
            
            # Clear intermediate tensors
            del inputs, outputs, depth
            
    except Exception as e:
        print(f"  Error during depth prediction: {e}")
        return None
    
    # Clear CUDA cache after processing
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    depth_img = Image.fromarray(depth_np)
    
    # Resize back to original size if we resized
    if depth_img.size != original_size:
        depth_img = depth_img.resize(original_size, Image.Resampling.LANCZOS)
        print(f"  Upscaled depth map back to {original_size}")
    
    processing_time = time.time() - start_time
    print(f"  Depth processing took {processing_time:.2f}s")
    
    return depth_img

def extract_test_frames(video_path: str, output_dir: str = "test_frames", num_frames: int = 5):
    """Extract frames from video."""
    Path(output_dir).mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
    
    # Calculate frame indices to extract (evenly spaced)
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    extracted_frames = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_path = os.path.join(output_dir, f"frame_{i:03d}_{frame_idx:06d}.png")
            
            # Save original frame
            Image.fromarray(frame_rgb).save(frame_path)
            extracted_frames.append((frame_path, frame_rgb))
            print(f"Extracted frame {i+1}/{num_frames}: {frame_path}")
        else:
            print(f"Warning: Could not read frame at index {frame_idx}")
    
    cap.release()
    return extracted_frames

def generate_depth_maps_optimized(extracted_frames, output_dir: str = "test_frames"):
    """Generate depth maps with CUDA optimization and progress tracking."""
    depth_frames = []
    total_frames = len(extracted_frames)
    
    print(f"Processing {total_frames} frames for depth estimation...")
    
    # Batch processing can be more efficient on CUDA
    for i, (frame_path, frame_rgb) in enumerate(extracted_frames):
        print(f"\n[{i+1}/{total_frames}] Processing: {os.path.basename(frame_path)}")
        
        # Convert to PIL Image
        pil_img = Image.fromarray(frame_rgb)
        print(f"  Image size: {pil_img.size}")
        
        # Show GPU memory usage
        if DEVICE.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU memory: {memory_allocated:.1f}GB allocated, {memory_cached:.1f}GB cached")
        
        # Generate depth map
        try:
            depth_img = predict_depth_optimized(pil_img)
            
            if depth_img is None:
                print(f"  Skipping frame due to processing error")
                continue
                
            # Save depth map
            depth_path = frame_path.replace(".png", "_depth.png")
            depth_img.save(depth_path)
            depth_frames.append((depth_path, depth_img))
            
            print(f"  ✓ Saved depth map: {os.path.basename(depth_path)}")
            
        except Exception as e:
            print(f"  ✗ Failed to process frame: {e}")
            continue
    
    return depth_frames

def create_side_by_side_comparison(extracted_frames, depth_frames, output_dir: str = "test_frames"):
    """Create comparison images."""
    print(f"\nCreating {len(depth_frames)} comparison images...")
    
    for i, ((orig_path, orig_frame), (depth_path, depth_img)) in enumerate(zip(extracted_frames, depth_frames)):
        if isinstance(orig_frame, np.ndarray):
            orig_img = Image.fromarray(orig_frame)
        else:
            orig_img = orig_frame
        
        # Create side-by-side image
        width, height = orig_img.size
        comparison = Image.new('RGB', (width * 2, height))
        
        # Paste original on left
        comparison.paste(orig_img, (0, 0))
        
        # Convert depth to RGB for pasting
        depth_rgb = depth_img.convert('RGB')
        comparison.paste(depth_rgb, (width, 0))
        
        # Save comparison
        comparison_path = os.path.join(output_dir, f"comparison_{i:03d}.png")
        comparison.save(comparison_path)
        print(f"  ✓ {os.path.basename(comparison_path)}")

def test_single_frame_first():
    """Test depth generation on a single small image first to verify CUDA setup."""
    print("\n=== CUDA Test with synthetic image ===")
    
    # Create a test image
    test_img = Image.new('RGB', (512, 512), color='blue')
    
    # Add some pattern
    pixels = test_img.load()
    for i in range(512):
        for j in range(512):
            pixels[i, j] = (i % 256, j % 256, (i + j) % 256)
    
    print("Testing depth prediction on 512x512 test image...")
    start_time = time.time()
    
    try:
        depth_result = predict_depth_optimized(test_img, max_size=512)
        
        if depth_result:
            test_time = time.time() - start_time
            print(f"✓ CUDA test successful! Processing took {test_time:.2f}s")
            
            # Save test result
            Path("test_frames").mkdir(exist_ok=True)
            test_img.save("test_frames/cuda_test_input.png")
            depth_result.save("test_frames/cuda_test_depth.png")
            print("✓ Test images saved to test_frames/")
            return True
        else:
            print("✗ Test failed - no depth result")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    video_path = "test_depth_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        exit(1)
    
    # First, test CUDA setup
    if not test_single_frame_first():
        print("CUDA test failed. Check your GPU setup.")
        exit(1)
    
    print("\n" + "="*60)
    print("CUDA test passed! Processing video frames with RTX 3080...")
    print("="*60)
    
    start_total = time.time()
    
    print("\n=== Extracting test frames ===")
    extracted_frames = extract_test_frames(video_path, num_frames=5)  # Process 5 frames
    
    if not extracted_frames:
        print("No frames extracted. Exiting.")
        exit(1)
    
    print("\n=== Generating depth maps with CUDA ===")
    depth_frames = generate_depth_maps_optimized(extracted_frames)
    
    if not depth_frames:
        print("No depth maps generated. Check for errors above.")
        exit(1)
    
    print("\n=== Creating side-by-side comparisons ===")
    create_side_by_side_comparison(extracted_frames, depth_frames)
    
    total_time = time.time() - start_total
    
    print(f"\n" + "="*60)
    print(f"✓ CUDA processing complete! Total time: {total_time:.2f}s")
    print(f"✓ Processed {len(depth_frames)} frames successfully")
    print(f"✓ Average time per frame: {total_time/len(depth_frames):.2f}s")
    
    if DEVICE.type == "cuda":
        print(f"✓ Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.1f}GB")
    
    print(f"✓ Check the 'test_frames' directory for results")
    print("="*60) 