# stage1_eval_roi.py
import os
import json
import numpy as np

def main():
    test_y_path = "data/np_data/test_y.npy"
    out_dir = "stage1_outputs"

    y_test = np.load(test_y_path)  # (N,H,W)

    results = []
    for i in range(y_test.shape[0]):
        gt = y_test[i].astype(bool)
        roi = np.load(os.path.join(out_dir, f"tile_{i:03d}_roi.npy")).astype(bool)

        gt_pixels = int(gt.sum())
        roi_pixels = int(roi.sum())
        total_pixels = gt.size

        if gt_pixels == 0:
            coverage = 1.0  # nothing to cover
        else:
            coverage = float((roi & gt).sum()) / float(gt_pixels)

        roi_frac = float(roi_pixels) / float(total_pixels)

        results.append({
            "tile": i,
            "gt_pixels": gt_pixels,
            "roi_pixels": roi_pixels,
            "coverage": coverage,
            "roi_frac": roi_frac,
        })

    # summary
    coverages = [r["coverage"] for r in results]
    roi_fracs = [r["roi_frac"] for r in results]

    summary = {
        "mean_coverage": float(np.mean(coverages)),
        "median_coverage": float(np.median(coverages)),
        "mean_roi_frac": float(np.mean(roi_fracs)),
        "median_roi_frac": float(np.median(roi_fracs)),
        "tiles": results,
    }

    print(json.dumps({
        "mean_coverage": summary["mean_coverage"],
        "median_coverage": summary["median_coverage"],
        "mean_roi_frac": summary["mean_roi_frac"],
        "median_roi_frac": summary["median_roi_frac"],
    }, indent=2))

    with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved eval_summary.json in", out_dir)

if __name__ == "__main__":
    main()
