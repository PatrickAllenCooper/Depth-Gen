# Depth-Gen

Local FastAPI server for Apple **Depth Pro** monocular depth estimation, optimised for Apple Silicon (M-series) GPUs via PyTorch MPS.

## Features

* 🔍 Zero-shot metric depth estimation using `apple/DepthPro-hf`.
* 🚀 Runs on M-series GPU (MPS), CUDA or CPU.
* 🖼️ Accepts JPEG/PNG image via `/depth` endpoint and returns 8-bit PNG depth map.
* 🩺 `/health` endpoint to verify service and device.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure you have PyTorch compiled with MPS support (default wheels for macOS ≥13 with arm64).

## Running

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

### Depth Endpoint

```bash
curl -X POST -F "file=@input.jpg" http://localhost:8000/depth -o depth.png -D headers.txt
```

The response PNG `depth.png` contains the normalized depth map. HTTP headers include `X-Field-Of-View` and `X-Focal-Length` for additional info.

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "device": "mps"}
```

## Environment variables

* `DEPTH_MODEL_REPO` – override Hugging Face repo id (default `apple/DepthPro-hf`).

## License

This project wraps the model released under Apple-ASCL. Code itself is MIT-licensed. 