# stage1_sweep_configs.py
import os, json
import numpy as np
from scipy import ndimage as ndi
import joblib
from itertools import product


def prob_map_for_tile(scaler, clf, tile_x):
    H, W, C = tile_x.shape
    X = tile_x.reshape(-1, C).astype(np.float32, copy=False)
    p = clf.predict_proba(scaler.transform(X))[:, 1]
    return p.reshape(H, W)


def roi_from_prob(prob_map, mode, threshold, keep_frac, min_blob_area, dilate_pixels, box_pad):
    """
    mode:
      - "fixed": prob >= threshold
      - "topk":  prob >= max(threshold, quantile(1-keep_frac))
      - "hybrid_or": (prob >= threshold) OR (prob >= quantile(1-keep_frac))
    Returns roi_mask uint8 {0,1}
    """
    H, W = prob_map.shape

    if mode == "fixed":
        mask = (prob_map >= threshold)

    else:
        # adaptive threshold based on keeping top keep_frac
        t_adapt = float(np.quantile(prob_map, 1.0 - keep_frac))
        if mode == "topk":
            t = max(float(threshold), t_adapt)
            mask = (prob_map >= t)
        elif mode == "hybrid_or":
            mask = (prob_map >= float(threshold)) | (prob_map >= t_adapt)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # dilation
    if dilate_pixels > 0:
        struct = ndi.generate_binary_structure(2, 1)
        mask = ndi.binary_dilation(mask, structure=struct, iterations=dilate_pixels)

    labeled, n = ndi.label(mask)
    if n == 0:
        return np.zeros((H, W), dtype=np.uint8)

    roi = np.zeros((H, W), dtype=np.uint8)
    slices = ndi.find_objects(labeled)
    for comp_id, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        comp = (labeled[slc] == comp_id)
        area = int(comp.sum())
        if area < min_blob_area:
            continue
        roi[slc][comp] = 1

    # Note: box_pad affects boxes not roi_frac/coverage; kept for completeness
    return roi


def eval_config(prob_maps, y_test, cfg):
    """
    Returns aggregate metrics + per-tile coverage list for robust stats.
    """
    N = y_test.shape[0]
    coverages = []
    roi_fracs = []

    for i in range(N):
        gt = y_test[i].astype(bool)
        prob = prob_maps[i]
        roi = roi_from_prob(prob, **cfg).astype(bool)

        gt_pixels = int(gt.sum())
        if gt_pixels == 0:
            coverage = 1.0
        else:
            coverage = float((roi & gt).sum()) / float(gt_pixels)

        roi_frac = float(roi.sum()) / float(roi.size)

        coverages.append(coverage)
        roi_fracs.append(roi_frac)

    coverages = np.array(coverages, dtype=np.float32)
    roi_fracs = np.array(roi_fracs, dtype=np.float32)

    out = {
        "mean_coverage": float(coverages.mean()),
        "median_coverage": float(np.median(coverages)),
        "p10_coverage": float(np.quantile(coverages, 0.10)),
        "mean_roi_frac": float(roi_fracs.mean()),
        "median_roi_frac": float(np.median(roi_fracs)),
        "p90_roi_frac": float(np.quantile(roi_fracs, 0.90)),
        "cfg": cfg,
    }
    return out


def pareto_frontier(results):
    """
    Keep configs that are not dominated:
      better coverage AND lower ROI at the same time.
    """
    # Sort by coverage desc, ROI asc
    r = sorted(results, key=lambda x: (-x["mean_coverage"], x["mean_roi_frac"]))
    frontier = []
    best_roi = float("inf")
    for item in r:
        roi = item["mean_roi_frac"]
        if roi < best_roi:
            frontier.append(item)
            best_roi = roi
    return frontier


def main():
    test_x_path = "data/np_data/test_x.npy"
    test_y_path = "data/np_data/test_y.npy"
    model_path  = "stage1_model_streaming/sgd_logreg.joblib"

    cache_dir = "stage1_sweep_cache"
    os.makedirs(cache_dir, exist_ok=True)
    prob_cache_path = os.path.join(cache_dir, "test_prob_maps.npy")

    print("Loading model:", model_path)
    bundle = joblib.load(model_path)
    scaler, clf = bundle["scaler"], bundle["clf"]

    print("Loading test tiles (mmap)...")
    X_test = np.load(test_x_path, mmap_mode="r")   # (N,512,512,16)
    y_test = np.load(test_y_path, mmap_mode="r")   # (N,512,512)

    # ---- Compute prob maps once and cache ----
    if os.path.exists(prob_cache_path):
        print("Loading cached prob maps:", prob_cache_path)
        prob_maps = np.load(prob_cache_path, mmap_mode="r")
    else:
        print("Computing prob maps (one-time)...")
        prob_maps = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2]), dtype=np.float32)
        for i in range(X_test.shape[0]):
            prob_maps[i] = prob_map_for_tile(scaler, clf, X_test[i]).astype(np.float32)
            if (i + 1) % 5 == 0:
                print(f"  computed {i+1}/{X_test.shape[0]}")
        np.save(prob_cache_path, prob_maps)
        print("Saved prob cache:", prob_cache_path)

    # ---- Sweep space ----
    # Start moderately sized; expand if needed.
    modes = ["fixed", "topk", "hybrid_or"]
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45]
    keep_fracs = [0.03, 0.05, 0.08, 0.12, 0.18]     # only used for modes != fixed
    min_blob_areas = [50, 150, 300, 600, 1200]
    dilates = [0, 1, 2]
    box_pads = [2, 4]  # doesn't affect metrics, but keep

    results = []
    total = 0

    for mode in modes:
        for threshold, min_blob_area, dilate_pixels, box_pad in product(thresholds, min_blob_areas, dilates, box_pads):
            if mode == "fixed":
                cfg = dict(
                    mode=mode,
                    threshold=threshold,
                    keep_frac=0.05,  # unused
                    min_blob_area=min_blob_area,
                    dilate_pixels=dilate_pixels,
                    box_pad=box_pad,
                )
                results.append(eval_config(prob_maps, y_test, cfg))
                total += 1
            else:
                for keep_frac in keep_fracs:
                    cfg = dict(
                        mode=mode,
                        threshold=threshold,
                        keep_frac=keep_frac,
                        min_blob_area=min_blob_area,
                        dilate_pixels=dilate_pixels,
                        box_pad=box_pad,
                    )
                    results.append(eval_config(prob_maps, y_test, cfg))
                    total += 1

        print(f"Finished mode={mode}")

    print("Total configs evaluated:", total)

    # ---- Save all results ----
    all_path = os.path.join(cache_dir, "sweep_results.json")
    with open(all_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", all_path)

    # ---- Report best by constraints ----
    # Pick your constraint for reporting:
    target_cov = 0.97
    candidates = [r for r in results if r["mean_coverage"] >= target_cov]
    candidates = sorted(candidates, key=lambda r: r["mean_roi_frac"])
    print(f"\nBest configs with mean_coverage >= {target_cov}:")
    for r in candidates[:10]:
        print({
            "mean_coverage": round(r["mean_coverage"], 4),
            "p10_coverage": round(r["p10_coverage"], 4),
            "mean_roi_frac": round(r["mean_roi_frac"], 4),
            "cfg": r["cfg"],
        })

    # Alternatively: ROI budget constraint
    roi_budget = 0.10
    candidates2 = [r for r in results if r["mean_roi_frac"] <= roi_budget]
    candidates2 = sorted(candidates2, key=lambda r: -r["mean_coverage"])
    print(f"\nBest configs with mean_roi_frac <= {roi_budget}:")
    for r in candidates2[:10]:
        print({
            "mean_coverage": round(r["mean_coverage"], 4),
            "p10_coverage": round(r["p10_coverage"], 4),
            "mean_roi_frac": round(r["mean_roi_frac"], 4),
            "cfg": r["cfg"],
        })

    # ---- Pareto frontier ----
    frontier = pareto_frontier(results)
    front_path = os.path.join(cache_dir, "pareto_frontier.json")
    with open(front_path, "w") as f:
        json.dump(frontier, f, indent=2)
    print("\nSaved Pareto frontier:", front_path)
    print("Top 10 Pareto points (coverage vs ROI):")
    for r in frontier[:10]:
        print(round(r["mean_coverage"], 4), round(r["mean_roi_frac"], 4), r["cfg"])


if __name__ == "__main__":
    main()
