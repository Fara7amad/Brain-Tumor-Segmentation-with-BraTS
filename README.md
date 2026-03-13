# Brain Tumor Segmentation with BraTS2020

A full-stack 3D U-Net implementation for brain tumor segmentation.

**Stack:** Python · NumPy · PyTorch · nibabel · FastAPI · React  
**Dataset:** [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/) — 369 training cases, 125 validation cases

---

## Dataset Structure

```
MICCAI_BraTS2020_TrainingData/
  BraTS20_Training_001/
    BraTS20_Training_001_flair.nii   ← FLAIR modality
    BraTS20_Training_001_t1.nii      ← T1-weighted
    BraTS20_Training_001_t1ce.nii    ← T1 contrast-enhanced
    BraTS20_Training_001_t2.nii      ← T2-weighted
    BraTS20_Training_001_seg.nii     ← Ground truth mask
  BraTS20_Training_002/ ...
  BraTS20_Training_369/
  name_mapping.csv
  survival_info.csv
```

Each volume: shape `(240, 240, 155)`, voxel size `1mm³`, dtype `float64`.

---
Check out my beginner-friendly explanation for the dataset: [Understanding and Processing the Brain Tumor Segmentation (BraTS2020) Dataset](https://medium.com/@farahabuhamad/understanding-and-processing-the-brain-tumor-segmentation-brats2020-dataset-98b67d303336)
---

## Key Data Facts

| Property | Value | Implication |
|---|---|---|
| Volume shape | `(240, 240, 155)` | 3D — needs 3D-aware model |
| Voxel size | `1mm × 1mm × 1mm` | Physical scale known |
| Non-zero fraction | ~15% | Crop before feeding to model |
| Intensity ranges | T2: 0–376, T1ce: 0–1845 | Normalize per modality |
| Seg labels | `{0, 1, 2, 4}` | Remap `4 → 3` before training |
| Background fraction | 97.63% | Cross-entropy alone won't work |

### MRI Modalities

| Modality | Measures | Key use |
|---|---|---|
| T1 | Longitudinal relaxation | Anatomy, grey/white matter |
| T1ce | T1 + gadolinium contrast | Active tumor (enhancing) |
| T2 | Transverse relaxation | Fluid, edema |
| FLAIR | T2 with free water suppressed | Peritumoral edema |

### BraTS Evaluation Regions

| Region | Labels | Clinical meaning |
|---|---|---|
| Whole Tumor (WT) | {1, 2, 3} | Total tumor extent |
| Tumor Core (TC) | {1, 3} | Surgically targetable core |
| Enhancing Tumor (ET) | {3} | Active, contrast-enhancing |

---

## Preprocessing Pipeline

```
Raw NIfTI (240×240×155)
        ↓
normalize_modality()     Z-score within brain mask, per modality
        ↓
crop_to_brain()          Tight bounding box of non-zero voxels
        ↓
resize_volume()          Trilinear interpolation → (128, 128, 128)
        ↓
Model input tensor       (4, 128, 128, 128) — 4 modalities stacked
```

### normalize_modality

```python
def normalize_modality(vol: np.ndarray) -> np.ndarray:
    brain_mask = vol > 0
    if brain_mask.sum() == 0:
        return vol
    mu  = vol[brain_mask].mean()
    std = vol[brain_mask].std() + 1e-8
    normalized = np.zeros_like(vol)
    normalized[brain_mask] = (vol[brain_mask] - mu) / std
    return normalized.astype(np.float32)
```

**Why:** MRI intensities are scanner-dependent — not comparable across patients or modalities.
Z-score within brain voxels gives each modality mean≈0, std≈1, regardless of scanner.

### crop_to_brain

```python
def crop_to_brain(vol: np.ndarray) -> np.ndarray:
    coords = np.array(np.where(vol > 0))
    if coords.shape[1] == 0:
        return vol
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1) + 1
    return vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
```

**Why:** 85% of the `(240,240,155)` volume is background air.
Cropping to the brain bounding box removes wasted computation and GPU memory.

### resize_volume

```python
def resize_volume(vol: np.ndarray, target=(128, 128, 128)) -> np.ndarray:
    tensor  = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target, mode="trilinear", align_corners=True)
    return resized.squeeze().numpy()
```

**Why:** The model requires fixed-size input. `128³` fits in ~10GB VRAM at batch size 1.
`align_corners=True` ensures image and mask stay spatially aligned when resized separately.

---

## Segmentation Label Remapping

BraTS2020 raw labels: `{0, 1, 2, 4}` — no label 3 (historical artifact).  
Before training, remap `4 → 3` so model output indices match:

```python
seg[seg == 4] = 3
# 0 = background
# 1 = necrotic core (NCR)
# 2 = peritumoral edema (ED)
# 3 = enhancing tumor (ET)  ← was label 4 in raw file
```

---

## Repository Structure

```
Brain-Tumor-Segmentation-with-BraTS/
├── README.md
├── src/
│   ├── dataset.py               ← BraTSDataset             
│   ├── model.py                 ← 3D U-Net                 
│   ├── train.py                 ← training loop            
│   └── inference.py             ← inference + export              
└── frontend/                    ← React UI                 
```

---

## References

- [BraTS 2020 Challenge](https://www.med.upenn.edu/cbica/brats2020/)
- [3D U-Net (Çiçek et al., 2016)](https://arxiv.org/abs/1606.06650)
- [nnU-Net (Isensee et al., 2021)](https://www.nature.com/articles/s41592-020-01008-z)
- [Dice Loss for medical segmentation](https://arxiv.org/abs/1707.03237)
---

---
## Screenshots

<img src= "">







---
