#!/usr/bin/env python3
"""
Full video depth map processor for Windows with CUDA optimization.
Processes entire videos and outputs depth map videos.
"""

import os
import cv2
import torch
import gc
import argparse
from PIL import Image
import numpy as np
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
from pathlib import Path
import time
from tqdm import tqdm

# Completely disable torch compilation
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Disable torch.compile globally
original_compile = torch.compile
torch.compile = lambda *args, **kwargs: args[0]  # Return model unchanged

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

class VideoDepthProcessor:
    def __init__(self, model_repo: str = "apple/DepthPro-hf"):
        self.model_repo = model_repo
        self.image_processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load model with CUDA optimizations"""
        print(f"Loading model from: {self.model_repo}")
        
        try:
            print("Loading image processor...")
            self.image_processor = DepthProImageProcessorFast.from_pretrained(self.model_repo)
            
            print("Loading depth estimation model...")
            self.model = DepthProForDepthEstimation.from_pretrained(
                self.model_repo,
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(DEVICE)
            
            self.model.eval()
            
            # CUDA-specific optimizations
            if DEVICE.type == "cuda":
                print("Applying CUDA optimizations...")
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Skip model compilation for now due to Triton dependency issues
                print("Skipping model compilation (using regular CUDA mode)")
            
            print("Model loaded and optimized successfully!")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def predict_depth(self, img: Image.Image, max_size: int = 1536) -> np.ndarray:
        """Generate depth map from PIL Image"""
        original_size = img.size
        
        # Resize if too large
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Clear cache
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        
        try:
            # Use autocast for CUDA mixed precision, but disable it if there are issues
            use_autocast = DEVICE.type == "cuda"
            
            if use_autocast:
                context = torch.autocast(device_type=DEVICE.type, dtype=torch.float16)
            else:
                context = torch.no_grad()
            
            with context:
                inputs = self.image_processor(images=img, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                post_processed = self.image_processor.post_process_depth_estimation(
                    outputs, target_sizes=[(img.height, img.width)]
                )[0]
                
                depth = post_processed["predicted_depth"]
                
                # Normalize to 0-255 range with safety checks
                if depth.numel() == 0:
                    raise ValueError("Empty depth tensor")
                    
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max > depth_min:
                    depth = (depth - depth_min) / (depth_max - depth_min) * 255.0
                else:
                    depth = torch.zeros_like(depth) + 128.0  # Mid-gray fallback
                    
                depth = depth.clip(0, 255).to(torch.uint8)
                depth_np = depth.detach().cpu().numpy()
                
                # Clear intermediate tensors
                del inputs, outputs, depth
                
        except Exception as e:
            print(f"Error during depth prediction: {e}")
            # Try without autocast as fallback
            if use_autocast and "autocast" not in str(e).lower():
                try:
                    with torch.no_grad():
                        inputs = self.image_processor(images=img, return_tensors="pt").to(DEVICE)
                        outputs = self.model(**inputs)
                        post_processed = self.image_processor.post_process_depth_estimation(
                            outputs, target_sizes=[(img.height, img.width)]
                        )[0]
                        depth = post_processed["predicted_depth"]
                        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                        depth = depth.clip(0, 255).to(torch.uint8)
                        depth_np = depth.detach().cpu().numpy()
                        del inputs, outputs, depth
                        print(f"Fallback without autocast succeeded for frame")
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
                    return None
            else:
                return None
        
        # Clear cache
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Resize back if we resized
        if img.size != original_size:
            depth_img = Image.fromarray(depth_np)
            depth_img = depth_img.resize(original_size, Image.Resampling.LANCZOS)
            depth_np = np.array(depth_img)
        
        return depth_np
    
    def process_video(self, input_path: str, output_path: str, 
                     max_size: int = 1536, batch_size: int = 1,
                     output_format: str = 'side_by_side'):
        """
        Process entire video and create depth map video.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            max_size: Maximum image dimension for processing
            batch_size: Number of frames to process at once (currently 1 for memory safety)
            output_format: 'depth_only', 'side_by_side', or 'original_only'
        """
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Input video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video writer
        if output_format == 'side_by_side':
            out_width, out_height = width * 2, height
        else:
            out_width, out_height = width, height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        if not out.isOpened():
            cap.release()
            raise ValueError(f"Could not create output video: {output_path}")
        
        print(f"Output video: {out_width}x{out_height}")
        print(f"Processing {total_frames} frames...")
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Generate depth map with retry logic
                depth_np = None
                max_retries = 3
                
                for retry in range(max_retries):
                    try:
                        depth_np = self.predict_depth(pil_img, max_size=max_size)
                        if depth_np is not None:
                            break
                        else:
                            print(f"Retry {retry + 1}/{max_retries} for frame {frame_count}")
                    except Exception as e:
                        print(f"Retry {retry + 1}/{max_retries} failed for frame {frame_count}: {e}")
                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(0.1)  # Brief pause before retry
                
                # If all retries failed, create a fallback depth map (black/zero depth)
                if depth_np is None:
                    print(f"Creating fallback depth map for frame {frame_count}")
                    depth_np = np.zeros((height, width), dtype=np.uint8)
                
                # Create output frame based on format
                if output_format == 'depth_only':
                    # Convert single channel depth to 3-channel grayscale
                    depth_3ch = cv2.cvtColor(depth_np, cv2.COLOR_GRAY2BGR)
                    output_frame = depth_3ch
                
                elif output_format == 'side_by_side':
                    # Convert depth to 3-channel
                    depth_3ch = cv2.cvtColor(depth_np, cv2.COLOR_GRAY2BGR)
                    # Combine original and depth side by side
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    output_frame = np.hstack([frame_bgr, depth_3ch])
                
                else:  # original_only
                    output_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(output_frame)
                
                frame_count += 1
                pbar.update(1)
                
                # Progress update every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
                    pbar.set_postfix({
                        'FPS': f'{fps_current:.1f}',
                        'ETA': f'{eta/60:.1f}min' if eta > 60 else f'{eta:.0f}s'
                    })
                    
                    if DEVICE.type == "cuda":
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        pbar.set_postfix({
                            'FPS': f'{fps_current:.1f}',
                            'ETA': f'{eta/60:.1f}min' if eta > 60 else f'{eta:.0f}s',
                            'GPU': f'{memory_used:.1f}GB'
                        })
        
        # Cleanup
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print(f"\n✓ Video processing complete!")
        print(f"✓ Processed {frame_count} frames in {total_time:.1f}s")
        print(f"✓ Average processing speed: {avg_fps:.1f} FPS")
        print(f"✓ Output saved to: {output_path}")
        
        if DEVICE.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"✓ Peak GPU memory usage: {peak_memory:.1f}GB")

def main():
    parser = argparse.ArgumentParser(description="Process video for depth estimation")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path (default: input_depth.mp4)")
    parser.add_argument("--max-size", type=int, default=1536, 
                       help="Maximum image dimension for processing (default: 1536)")
    parser.add_argument("--format", choices=['depth_only', 'side_by_side', 'original_only'],
                       default='side_by_side',
                       help="Output format (default: side_by_side)")
    parser.add_argument("--model", default="apple/DepthPro-hf",
                       help="Model repository (default: apple/DepthPro-hf)")
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_depth{input_path.suffix}")
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return 1
    
    try:
        # Create processor and process video
        processor = VideoDepthProcessor(model_repo=args.model)
        processor.process_video(
            input_path=args.input,
            output_path=args.output,
            max_size=args.max_size,
            output_format=args.format
        )
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 