import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path

MODALITIES   = ["flair", "t1", "t1ce", "t2"]
TARGET_SHAPE = (128, 128, 128)


# ─── Preprocessing Functions ──────────────────────────────────────────────────

def normalize_modality(vol: np.ndarray) -> np.ndarray:
    brain_mask = vol > 0
    if brain_mask.sum() == 0:
        return vol
    mu  = vol[brain_mask].mean()
    std = vol[brain_mask].std() + 1e-8
    normalized = np.zeros_like(vol)
    normalized[brain_mask] = (vol[brain_mask] - mu) / std
    return normalized.astype(np.float32)


def crop_to_brain(vol: np.ndarray) -> np.ndarray:
    coords = np.array(np.where(vol > 0))
    if coords.shape[1] == 0:
        return vol
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1) + 1
    return vol[mins[0]:maxs[0],
               mins[1]:maxs[1],
               mins[2]:maxs[2]]


def resize_volume(vol: np.ndarray, target=TARGET_SHAPE,
                  mode="trilinear") -> np.ndarray:
    tensor = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    kwargs = {"align_corners": True} if mode == "trilinear" else {}
    resized = F.interpolate(tensor, size=target, mode=mode, **kwargs)
    return resized.squeeze().numpy()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS2020 training data.

    Returns per sample:
        images : float32 tensor  (4, 128, 128, 128)  — 4 modalities
        mask   : long tensor     (128, 128, 128)      — labels {0,1,2,3}
    """

    def __init__(self, root_dir: str, split: str = "train",
                 train_ratio: float = 0.8, seed: int = 42):
      
        root = Path(root_dir)
        cases = sorted([d for d in root.iterdir() if d.is_dir()])
        rng = np.random.default_rng(seed)
        rng.shuffle(cases)

        # Split into train / val
        n_train = int(len(cases) * train_ratio)
        if split == "train":
            self.cases = cases[:n_train]
        else:
            self.cases = cases[n_train:]

        self.split = split

    def __len__(self):
        # DataLoader calls this to know how many batches to produce per epoch
        return len(self.cases)

    def __getitem__(self, idx: int):
    
        case_dir = self.cases[idx]
        case_id  = case_dir.name

        # ── Load and preprocess all 4 modalities ─────────────────────────────
        volumes = []
        for mod in MODALITIES:
            path = case_dir / f"{case_id}_{mod}.nii"
            vol  = nib.load(str(path)).get_fdata().astype(np.float32)
            vol  = normalize_modality(vol)
            vol  = crop_to_brain(vol)
            vol  = resize_volume(vol, mode="trilinear")
            volumes.append(vol)

        # Stack: list of 4 × (128,128,128) → (4, 128, 128, 128)
        stacked = np.stack(volumes, axis=0)

        # ── Load and preprocess segmentation mask ─────────────────────────────
        seg_path = case_dir / f"{case_id}_seg.nii"
        seg = nib.load(str(seg_path)).get_fdata().astype(np.uint8)

        seg[seg == 4] = 3                              # remap label 4 → 3
        seg = resize_volume(seg, mode="nearest")       # nearest for labels
        seg = seg.astype(np.int64)

        # ── Convert to tensors ────────────────────────────────────────────────
        images = torch.from_numpy(stacked).float()    # (4, 128, 128, 128)
        mask   = torch.from_numpy(seg).long()         # (128, 128, 128)

        return images, mask