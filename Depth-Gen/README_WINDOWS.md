# Depth-Gen for Windows with CUDA

Windows-optimized version of the Depth-Gen project for NVIDIA RTX GPUs, specifically tested on RTX 3080.

## Features

* 🔍 Zero-shot metric depth estimation using Apple's DepthPro model
* 🚀 CUDA-optimized for NVIDIA RTX GPUs (3070, 3080, 3090, 4070, 4080, 4090)
* 🎥 Full video processing with progress tracking
* 🖼️ Single image processing via FastAPI server
* 🔧 Mixed precision support for faster inference on RTX cards
* 📊 Real-time GPU memory monitoring

## System Requirements

* Windows 10/11
* NVIDIA RTX GPU (RTX 20/30/40 series recommended)
* CUDA 11.8+ or 12.0+
* Python 3.8+
* 8GB+ GPU VRAM (12GB+ recommended for large videos)

## Installation

### 1. Install CUDA
Download and install CUDA from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)

### 2. Set up Python environment
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify CUDA installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Quick Test (Recommended First Step)
Test your setup with a few frames from your video:
```bash
python test_video_depth_cuda.py
```

This will:
- Test CUDA setup with a synthetic image
- Extract 5 frames from `test_depth_video.mp4`
- Generate depth maps for each frame
- Create side-by-side comparisons
- Show performance metrics

### Full Video Processing
Process entire videos to create depth map videos:

```bash
# Basic usage - creates side-by-side video
python video_depth_processor.py test_depth_video.mp4

# Specify output file
python video_depth_processor.py test_depth_video.mp4 -o output_depth.mp4

# Depth-only output
python video_depth_processor.py test_depth_video.mp4 --format depth_only

# Larger processing size (requires more VRAM)
python video_depth_processor.py test_depth_video.mp4 --max-size 2048
```

### FastAPI Server
Run the depth estimation server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Test with curl:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/depth -o depth.png
```

## Output Formats

* **side_by_side**: Original video on left, depth map on right (default)
* **depth_only**: Depth map only (grayscale converted to video)
* **original_only**: Original video (for testing)

## Performance Optimization

### RTX 3080 Specific
- Uses mixed precision (FP16) for 2x faster inference
- Enables TensorFloat-32 for additional speedup
- Model compilation when available
- Automatic memory management

### Expected Performance
- RTX 3080: ~2-5 FPS depending on video resolution
- RTX 4080/4090: ~3-8 FPS
- Processing time scales with video resolution and length

### Memory Management
- Automatic CUDA cache clearing
- Progressive image resizing for large inputs
- Memory usage monitoring and reporting

## Troubleshooting

### CUDA Out of Memory
1. Reduce `--max-size` parameter (try 1024 or 768)
2. Close other GPU-intensive applications
3. Use smaller batch sizes

### Slow Performance
1. Verify CUDA is being used: Look for "CUDA available!" message
2. Check GPU utilization with `nvidia-smi`
3. Ensure sufficient VRAM is available
4. Consider upgrading PyTorch for latest optimizations

### Model Loading Issues  
1. Check internet connection (models download from Hugging Face)
2. Clear Hugging Face cache: `~/.cache/huggingface/`
3. Verify transformers version: `pip show transformers`

## File Structure

```
Depth-Gen/
├── app/
│   └── main.py                 # FastAPI server (CUDA optimized)
├── test_video_depth_cuda.py    # CUDA test script
├── video_depth_processor.py    # Full video processor
├── requirements.txt            # Python dependencies
└── README_WINDOWS.md          # This file
```

## Advanced Usage

### Custom Model
Use a different DepthPro model:
```bash
python video_depth_processor.py input.mp4 --model "custom/depth-model"
```

### Batch Processing
Process multiple videos:
```bash
for video in *.mp4; do
    python video_depth_processor.py "$video"
done
```

### GPU Monitoring
Monitor real-time GPU usage:
```bash
# In separate terminal
nvidia-smi -l 1
```

## Performance Benchmarks

Tested on RTX 3080 (10GB VRAM):

| Resolution | FPS | Memory Usage | Notes |
|------------|-----|--------------|-------|
| 720p       | ~5  | ~3GB         | Optimal |
| 1080p      | ~3  | ~5GB         | Good |
| 1440p      | ~2  | ~7GB         | Max recommended |
| 4K         | ~1  | ~9GB         | Requires max_size reduction |

## Support

For Windows-specific issues:
1. Ensure CUDA drivers are up to date
2. Check PyTorch CUDA compatibility
3. Verify video codecs are installed
4. Monitor GPU temperatures under load

For general depth estimation questions, refer to the main README.md 