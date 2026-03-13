"""
inference.py — FastAPI Backend for BraTS Segmentation
=======================================================
Loads the trained UNet3D checkpoint and serves predictions via HTTP.

Endpoints:
    GET  /health              — model status
    POST /segment             — run segmentation on uploaded NIfTI files
    POST /segment/demo        — run on a synthetic volume (no upload needed)

Run:
    cd src
    uvicorn inference:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import sys
import numpy as np
import torch
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
load_dotenv()
# Make sure src/ is on the path when running from project root
sys.path.append(str(Path(__file__).parent))

from model   import UNet3D
from dataset import normalize_modality, crop_to_brain, resize_volume, MODALITIES


# ─── App Setup ────────────────────────────────────────────────────────────────
# CORSMiddleware allows the React frontend (running on localhost:5173)
# to call this API without being blocked by the browser's same-origin policy.

app = FastAPI(
    title="BraTS Segmentation API",
    description="3D U-Net brain tumor segmentation — BraTS2020",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Model Loading ────────────────────────────────────────────────────────────
# Model is loaded once at startup and reused for every request.
# Loading per-request would be ~5 seconds of overhead each time.

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = Path(__file__).parent.parent / os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pth")
TARGET     = (128, 128, 128)

model: UNet3D | None = None


@app.on_event("startup")
def load_model():
    global model
    model = UNet3D(in_channels=4, out_channels=4,
                   base_filters=32, depth=4).to(DEVICE)

    if CHECKPOINT.exists():
        ckpt = torch.load(str(CHECKPOINT), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']}  "
              f"best Dice: {ckpt['best_dice']:.4f}")
    else:
        print("⚠️  No checkpoint found — using random weights")

    model.eval()


# ─── Helpers ──────────────────────────────────────────────────────────────────
# Converts a raw NIfTI bytes object into a preprocessed numpy array.
# Supports .nii and .nii.gz — nibabel detects format from the header.

def load_nifti_bytes(content: bytes, filename: str) -> np.ndarray:
    try:
        import nibabel as nib
        import tempfile, os
        suffix = ".nii.gz" if filename.endswith(".gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        vol = nib.load(tmp_path).get_fdata().astype(np.float32)
        os.unlink(tmp_path)
        return vol
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load {filename}: {e}")


def preprocess_volume(volumes: list[np.ndarray]) -> torch.Tensor:
    # Apply full pipeline to each modality: normalize → crop → resize
    # Then stack → (1, 4, 128, 128, 128) with batch dim
    processed = []
    for vol in volumes:
        vol = normalize_modality(vol)
        vol = crop_to_brain(vol)
        vol = resize_volume(vol, target=TARGET, mode="trilinear")
        processed.append(vol)
    stacked = np.stack(processed, axis=0)                      # (4, 128, 128, 128)
    return torch.from_numpy(stacked).float().unsqueeze(0)      # (1, 4, 128, 128, 128)


def run_inference(input_tensor: torch.Tensor) -> np.ndarray:
    # Returns (128, 128, 128) integer label map {0,1,2,3}
    input_tensor = input_tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)                           # (1, 4, 128, 128, 128)
        pred   = torch.argmax(logits, dim=1).squeeze(0)       # (128, 128, 128)
    return pred.cpu().numpy().astype(np.uint8)


def build_response(pred: np.ndarray, volumes: list[np.ndarray] | None = None, demo: bool = False) -> dict:
    total = pred.size
    classes = {}
    class_names  = {0: "Background", 1: "Necrotic Core", 2: "Edema", 3: "Enhancing Tumor"}
    class_colors = {0: [0,0,0,0], 1: [255,50,20,200], 2: [0,220,80,200], 3: [255,220,0,200]}

    for label in range(4):
        count = int((pred == label).sum())
        classes[str(label)] = {
            "name":       class_names[label],
            "voxels":     count,
            "percentage": round(100 * count / total, 2),
            "color":      class_colors[label],
        }

    regions = {
        "WT": int((pred > 0).sum()),
        "TC": int(np.isin(pred, [1, 3]).sum()),
        "ET": int((pred == 3).sum()),
    }

    h, w, d = pred.shape

    # Segmentation slices
    slices = {
        "axial":    pred[:, :, d // 2].tolist(),
        "coronal":  pred[:, w // 2, :].tolist(),
        "sagittal": pred[h // 2, :, :].tolist(),
    }

    # MRI slices — normalize each modality to 0-255 for display
    # FLAIR (index 0) is best for showing tumor context
    mri_slices = {}
    if volumes is not None:
        flair = volumes[0]   # FLAIR is most informative for tumor visualization
        # Normalize to 0–255 for frontend rendering
        flair_min, flair_max = flair.min(), flair.max()
        flair_norm = ((flair - flair_min) / (flair_max - flair_min + 1e-8) * 255).astype(np.uint8)
        mri_slices = {
            "axial":    flair_norm[:, :, d // 2].tolist(),
            "coronal":  flair_norm[:, w // 2, :].tolist(),
            "sagittal": flair_norm[h // 2, :, :].tolist(),
        }

    return {
        "success":        True,
        "demo":           demo,
        "shape":          list(pred.shape),
        "tumor_burden_%": round(100 * (pred > 0).sum() / total, 3),
        "classes":        classes,
        "regions":        regions,
        "slices":         slices,
        "mri_slices":     mri_slices,
    }

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    # Called by frontend on load to check if the model is ready
    return {
        "status":           "ok",
        "device":           str(DEVICE),
        "model_loaded":     model is not None,
        "checkpoint_found": CHECKPOINT.exists(),
    }


@app.post("/segment")
async def segment(
    flair: UploadFile = File(...),
    t1:    UploadFile = File(...),
    t1ce:  UploadFile = File(...),
    t2:    UploadFile = File(...),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    uploads = [flair, t1, t1ce, t2]
    volumes = []
    for upload in uploads:
        content = await upload.read()
        vol     = load_nifti_bytes(content, upload.filename)
        volumes.append(vol)

    tensor = preprocess_volume(volumes)

    # Also get the preprocessed volumes for visualization
    preprocessed_vols = []
    for vol in volumes:
        v = normalize_modality(vol)
        v = crop_to_brain(v)
        v = resize_volume(v, target=TARGET, mode="trilinear")
        preprocessed_vols.append(v)

    pred = run_inference(tensor)
    return JSONResponse(build_response(pred, volumes=preprocessed_vols, demo=False))


@app.post("/segment/demo")
def segment_demo():
    # Runs inference on a synthetic random volume — no file upload needed.
    # Useful for testing the frontend without real patient data.
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    synthetic = torch.randn(1, 4, 128, 128, 128)
    pred      = run_inference(synthetic)
    return JSONResponse(build_response(pred, volumes=None, demo=True))