# src/preprocessing/build_hdf5.py
from __future__ import annotations
import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import h5py
import cv2

from src.config import load_config
from src.image_utils import load_image_set, extract_bounding_boxes
from src.constants import IMAGE_FILE_NAMES

def list_sample_dirs(root: Path, label_binary_name: str) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and (p/label_binary_name).exists()])

def to_chw(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return np.transpose(x, (2, 0, 1))

def mask_to_chw(m: np.ndarray) -> np.ndarray:
    if m.ndim == 2:
        m = m[..., None]
    m = (m > 0).astype(np.uint8)
    return np.transpose(m, (2, 0, 1))

def minmax01_inplace(x: np.ndarray):
    for c in range(x.shape[0]):
        ch = x[c]; mn, mx = float(ch.min()), float(ch.max())
        if mx > mn: ch[:] = (ch - mn) / (mx - mn)
        else: ch[:] = 0.0

def standardize_clip_inplace(x: np.ndarray, mean: float, std: float, cmin: float, cmax: float):
    x -= mean
    x /= (std + 1e-8)
    np.clip(x, cmin, cmax, out=x)

def resize_chw(x: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    C, H, W = x.shape
    oh, ow = out_hw
    y = np.empty((C, oh, ow), dtype=x.dtype)
    for c in range(C):
        y[c] = cv2.resize(x[c], (ow, oh), interpolation=cv2.INTER_LINEAR)
    return y

def write_h5_bbox(out_path: Path, imgs: list[np.ndarray], boxes5: list[np.ndarray], dirs: list[str]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not imgs:
        return
    C, H, W = imgs[0].shape
    with h5py.File(out_path, "w") as f:
        f.create_dataset("images", data=np.stack(imgs, 0).astype(np.float16),
                         compression="gzip", compression_opts=4, chunks=(1, C, H, W))
        f.create_dataset("bboxes", data=np.stack(boxes5, 0).astype(np.float32),
                         compression="gzip", compression_opts=4)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("dirs", data=np.array(dirs, dtype=object), dtype=dt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/defaults.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    dc = cfg["data"]

    raw_root = Path(dc["raw_root"])
    out_root = Path(dc["processed_root"]); out_root.mkdir(parents=True, exist_ok=True)
    Hb, Wb = dc["input_size"]["bbox"]
    label_binary_name = dc.get("label_binary_name", "labelbinary.tif")
    max_boxes = int(dc.get("max_boxes", 10))

    nc = dc["normalize"]; do_norm = bool(nc.get("enabled", True))
    mean, std = float(nc.get("mean", 0.5)), float(nc.get("std", 0.25))
    cmin, cmax = float(nc.get("clip_min", 0.0)), float(nc.get("clip_max", 1.0))
    sp = dc["split"]; train_ratio, seed = float(sp["train_ratio"]), int(sp["seed"])

    tiles = list_sample_dirs(raw_root, label_binary_name)
    assert tiles, f"No tiles under {raw_root} (looked for '{label_binary_name}')"

    rng = np.random.default_rng(seed)
    idx = np.arange(len(tiles)); rng.shuffle(idx)
    ntr = int(len(idx) * train_ratio)
    train_idx, val_idx = idx[:ntr], idx[ntr:]

    def process(indices):
        imgs, boxes, dirs, meta_rows = [], [], [], []
        for tid in indices:
            tdir = tiles[tid]
            try:
                img_hwc, mask_hw1 = load_image_set(tdir.as_posix(), tuple(IMAGE_FILE_NAMES))
            except Exception as e:
                print(f"[warn] skip {tdir}: {e}")
                continue

            H0, W0, C0 = img_hwc.shape
            assert C0 == len(IMAGE_FILE_NAMES), f"expected {len(IMAGE_FILE_NAMES)} channels, got {C0} @ {tdir}"

            x = to_chw(img_hwc)
            _ = mask_to_chw(mask_hw1)  # only to ensure binary label exists, boxes computed by your helper

            minmax01_inplace(x)
            if do_norm:
                standardize_clip_inplace(x, mean, std, cmin, cmax)

            x_resized = resize_chw(x, (Hb, Wb))

            boxes5 = extract_bounding_boxes(mask_hw1.squeeze().astype(np.uint8),
                                            num_boxes=max_boxes, force_square=False)
            if len(boxes5) > max_boxes:
                boxes5 = boxes5[:max_boxes]
            elif len(boxes5) < max_boxes:
                boxes5 += [(0.0, 0.0, 0.0, 0.0, 0.0)] * (max_boxes - len(boxes5))
            boxes5 = np.array(boxes5, dtype=np.float32)

            imgs.append(x_resized)
            boxes.append(boxes5)
            dirs.append(tdir.as_posix())
            meta_rows.append({"tile_idx": len(imgs)-1, "tile_dir": tdir.as_posix(),
                              "H0": H0, "W0": W0, "num_boxes": int((boxes5[:,0] > 0).sum())})
        return imgs, boxes, dirs, pd.DataFrame(meta_rows)

    tr_imgs, tr_boxes, tr_dirs, tr_meta = process(train_idx)
    va_imgs, va_boxes, va_dirs, va_meta = process(val_idx)

    write_h5_bbox(out_root / "bbox_train.h5", tr_imgs, tr_boxes, tr_dirs)
    write_h5_bbox(out_root / "bbox_val.h5",   va_imgs, va_boxes, va_dirs)
    tr_meta.to_csv(out_root / "bbox_train_meta.csv", index=False)
    va_meta.to_csv(out_root / "bbox_val_meta.csv",   index=False)
    print("[ok] Wrote bbox HDF5 + sidecars to", out_root)

if __name__ == "__main__":
    main()
