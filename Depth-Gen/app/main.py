import io
import os
from typing import Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

app = FastAPI(title="Depth-Gen: Apple Depth Pro Server", version="0.1.0")

# -----------------------------------------------------------------------------
# Device selection optimised for Apple Silicon.
# Prefer MPS if available (Apple GPU). Fallback to CUDA then CPU.
# -----------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# -----------------------------------------------------------------------------
# Model and processor are loaded once at startup.
# -----------------------------------------------------------------------------
MODEL_REPO = os.getenv("DEPTH_MODEL_REPO", "apple/DepthPro-hf")

try:
    image_processor: DepthProImageProcessorFast = DepthProImageProcessorFast.from_pretrained(MODEL_REPO)
    model: DepthProForDepthEstimation = DepthProForDepthEstimation.from_pretrained(MODEL_REPO).to(DEVICE)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load DepthPro model from {MODEL_REPO}: {e}") from e


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _predict_depth(img: Image.Image) -> Tuple[Image.Image, float, float]:
    """Run depth prediction and post-process.

    Returns
    -------
    depth_map_png : PIL.Image.Image
        Depth map scaled to 0-255 grayscale in uint8.
    fov : float
    focal_length_px : float
    """
    inputs = image_processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed = image_processor.post_process_depth_estimation(
        outputs, target_sizes=[(img.height, img.width)]
    )[0]

    depth = post_processed["predicted_depth"]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.clip(0, 255).to(torch.uint8)
    depth_np = depth.detach().cpu().numpy()
    depth_img = Image.fromarray(depth_np)

    return depth_img, post_processed["field_of_view"], post_processed["focal_length"]


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------

@app.post("/depth")
async def depth_from_image(file: UploadFile = File(...)):
    """Generate a depth map for a single RGB image.

    Parameters
    ----------
    file : UploadFile
        RGB image (PNG/JPEG).

    Returns
    -------
    StreamingResponse
        PNG image of depth map.
    """
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload PNG or JPEG.")

    try:
        bytes_data = await file.read()
        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    depth_img, fov, focal_length = _predict_depth(img)

    # Encode depth image to PNG bytes
    buf = io.BytesIO()
    depth_img.save(buf, format="PNG")
    buf.seek(0)

    headers = {
        "X-Field-Of-View": str(fov),
        "X-Focal-Length": str(focal_length),
    }

    return StreamingResponse(buf, media_type="image/png", headers=headers)


@app.get("/health")
async def health_check():
    """Simple health endpoint."""
    return JSONResponse({"status": "ok", "device": str(DEVICE)}) 