# Depth-Gen

Local FastAPI server and video processor for Apple **Depth Pro** monocular depth estimation, optimized for Apple Silicon (M-series) GPUs via PyTorch MPS and NVIDIA RTX GPUs via CUDA.

## Features

* 🔍 Zero-shot metric depth estimation using `apple/DepthPro-hf`
* 🚀 **Multi-platform support**: M-series GPU (MPS), NVIDIA CUDA, or CPU
* 🎥 **Full video processing** with robust error handling (Windows/CUDA)
* 🖼️ FastAPI server: JPEG/PNG image via `/depth` endpoint → 8-bit PNG depth map
* 🩺 `/health` endpoint to verify service and device
* ✅ **Production-tested** on 38,981-frame videos (RTX 3080)

## Installation

### macOS (Apple Silicon)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure you have PyTorch compiled with MPS support (default wheels for macOS ≥13 with arm64).

### Windows (NVIDIA CUDA)
For video processing with CUDA support, see [README_WINDOWS.md](README_WINDOWS.md) for detailed setup instructions.

**Quick start:**
```bash
conda create -n depth-gen-cuda python=3.11 -y
conda activate depth-gen-cuda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install fastapi uvicorn transformers accelerate opencv-python tqdm
```

## Running

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

### API Server

#### Depth Endpoint
```bash
curl -X POST -F "file=@input.jpg" http://localhost:8000/depth -o depth.png -D headers.txt
```

The response PNG `depth.png` contains the normalized depth map. HTTP headers include `X-Field-Of-View` and `X-Focal-Length` for additional info.

#### Health Check
```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "device": "mps"}
```

### Video Processing (Windows/CUDA)

#### Quick Test
```bash
python test_video_depth_cuda.py
```

#### Full Video Processing
```bash
# Side-by-side output (original + depth)
python video_depth_processor.py input.mp4 -o output_depth.mp4

# Depth-only output  
python video_depth_processor.py input.mp4 --format depth_only

# Custom output size
python video_depth_processor.py input.mp4 --max-size 1536
```

**Performance**: RTX 3080 processes 1080p video at ~1.4 FPS with robust error handling.

## Environment variables

* `DEPTH_MODEL_REPO` – override Hugging Face repo id (default `apple/DepthPro-hf`).

## License

This project wraps the model released under Apple-ASCL. Code itself is MIT-licensed. 