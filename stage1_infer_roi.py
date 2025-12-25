# # stage1_infer_roi.py
# import os
# import json
# import numpy as np
# from scipy import ndimage as ndi
# import joblib


# def prob_map_for_tile(model, tile_x):
#     """
#     tile_x: (H, W, C)
#     returns prob_map: (H, W)
#     """
#     H, W, C = tile_x.shape
#     X = tile_x.reshape(-1, C)
#     p = model.predict_proba(X)[:, 1]
#     return p.reshape(H, W)


# def roi_from_prob(prob_map, threshold=0.15, min_blob_area=100, dilate_pixels=1, box_pad=2):
#     """
#     Returns:
#       roi_mask: (H, W) uint8 0/1
#       boxes: list of dicts: {x1,y1,x2,y2,area}
#     Notes:
#       - threshold is chosen for high recall; tune later
#       - blob filtering removes speckle noise
#     """
#     H, W = prob_map.shape

#     mask = (prob_map >= threshold)

#     # dilate (optional)
#     if dilate_pixels > 0:
#         struct = ndi.generate_binary_structure(2, 1)
#         mask = ndi.binary_dilation(mask, structure=struct, iterations=dilate_pixels)

#     # connected components
#     labeled, n = ndi.label(mask)
#     if n == 0:
#         return np.zeros((H, W), dtype=np.uint8), []

#     boxes = []
#     roi_mask = np.zeros((H, W), dtype=np.uint8)

#     slices = ndi.find_objects(labeled)
#     for comp_id, slc in enumerate(slices, start=1):
#         if slc is None:
#             continue
#         comp = (labeled[slc] == comp_id)
#         area = int(comp.sum())
#         if area < min_blob_area:
#             continue

#         # add to final mask
#         roi_mask[slc][comp] = 1

#         y1, y2 = slc[0].start, slc[0].stop
#         x1, x2 = slc[1].start, slc[1].stop

#         # pad + clamp
#         x1p = max(0, x1 - box_pad)
#         y1p = max(0, y1 - box_pad)
#         x2p = min(W, x2 + box_pad)
#         y2p = min(H, y2 + box_pad)

#         boxes.append({"x1": x1p, "y1": y1p, "x2": x2p, "y2": y2p, "area": area})

#     return roi_mask, boxes


# def main():
#     test_x_path = "data/np_data/test_x.npy"
#     out_dir = "stage1_outputs"
#     os.makedirs(out_dir, exist_ok=True)

#     model = joblib.load("stage1_model/logreg.joblib")
#     X_test = np.load(test_x_path)  # (N,512,512,16)

#     # Tunable hyperparams
#     threshold = 0.35
#     min_blob_area = 350
#     dilate_pixels = 1
#     box_pad = 3

#     cfg = dict(
#         threshold=threshold,
#         min_blob_area=min_blob_area,
#         dilate_pixels=dilate_pixels,
#         box_pad=box_pad
#     )
#     with open(os.path.join(out_dir, "config.json"), "w") as f:
#         json.dump(cfg, f, indent=2)

#     for i in range(X_test.shape[0]):
#         tile = X_test[i]
#         prob = prob_map_for_tile(model, tile)
#         roi_mask, boxes = roi_from_prob(
#             prob,
#             threshold=threshold,
#             min_blob_area=min_blob_area,
#             dilate_pixels=dilate_pixels,
#             box_pad=box_pad
#         )

#         # save outputs
#         np.save(os.path.join(out_dir, f"tile_{i:03d}_prob.npy"), prob.astype(np.float32))
#         np.save(os.path.join(out_dir, f"tile_{i:03d}_roi.npy"), roi_mask.astype(np.uint8))
#         with open(os.path.join(out_dir, f"tile_{i:03d}_boxes.json"), "w") as f:
#             json.dump(boxes, f, indent=2)

#         print(f"tile {i:03d}: boxes={len(boxes)} roi_pixels={int(roi_mask.sum())}")

#     print("Done:", out_dir)


# if __name__ == "__main__":
#     main()


# stage1_infer_roi.py
import os
import json
import numpy as np
from scipy import ndimage as ndi
import joblib


def prob_map_for_tile(scaler, clf, tile_x):
    """
    tile_x: (H, W, C)
    returns prob_map: (H, W)
    """
    H, W, C = tile_x.shape
    X = tile_x.reshape(-1, C).astype(np.float32, copy=False)
    Xs = scaler.transform(X)
    p = clf.predict_proba(Xs)[:, 1]
    return p.reshape(H, W)


def roi_from_prob(
    prob_map,
    threshold=0.25,          # minimum threshold floor
    keep_frac=0.05,          # keep top 5% pixels per tile
    min_blob_area=1200,
    dilate_pixels=1,
    box_pad=4
):
    """
    Strategy:
      - adaptive threshold per tile: keep top keep_frac pixels
      - also enforce a floor threshold to avoid keeping pure noise
    """
    H, W = prob_map.shape

    # Adaptive threshold: keep top keep_frac pixels
    t_adapt = float(np.quantile(prob_map, 1.0 - keep_frac))
    t = max(float(threshold), t_adapt)

    mask = (prob_map >= t)

    # dilate (optional)
    if dilate_pixels > 0:
        struct = ndi.generate_binary_structure(2, 1)
        mask = ndi.binary_dilation(mask, structure=struct, iterations=dilate_pixels)

    labeled, n = ndi.label(mask)
    if n == 0:
        return np.zeros((H, W), dtype=np.uint8), []

    boxes = []
    roi_mask = np.zeros((H, W), dtype=np.uint8)

    slices = ndi.find_objects(labeled)
    for comp_id, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        comp = (labeled[slc] == comp_id)
        area = int(comp.sum())
        if area < min_blob_area:
            continue

        roi_mask[slc][comp] = 1

        y1, y2 = slc[0].start, slc[0].stop
        x1, x2 = slc[1].start, slc[1].stop

        x1p = max(0, x1 - box_pad)
        y1p = max(0, y1 - box_pad)
        x2p = min(W, x2 + box_pad)
        y2p = min(H, y2 + box_pad)

        boxes.append({"x1": x1p, "y1": y1p, "x2": x2p, "y2": y2p, "area": area})

    return roi_mask, boxes


def main():
    test_x_path = "data/np_data/test_x.npy"
    out_dir = "stage1_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Load streaming-trained model bundle
    bundle = joblib.load("stage1_model_streaming/sgd_logreg.joblib")
    scaler = bundle["scaler"]
    clf = bundle["clf"]

    # Use mmap to reduce RAM while loading test tiles
    X_test = np.load(test_x_path, mmap_mode="r")  # (N,512,512,16)

    # Tunable hyperparams
    threshold = 0.12       # floor (lets weak plumes in)
    keep_frac = 0.1       # cap ROI near 5% before blob filtering
    min_blob_area = 1200    # slightly lower because mask is sparser now
    dilate_pixels = 1      # reduce dilation to avoid giant merge
    box_pad = 4

    cfg = dict(
        model_path="stage1_model_streaming/sgd_logreg.joblib",
        threshold=threshold,
        min_blob_area=min_blob_area,
        dilate_pixels=dilate_pixels,
        box_pad=box_pad,
        keep_frac=keep_frac
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for i in range(X_test.shape[0]):
        tile = X_test[i]
        prob = prob_map_for_tile(scaler, clf, tile)
        roi_mask, boxes = roi_from_prob(
            prob,
            threshold=threshold,
            keep_frac=keep_frac,
            min_blob_area=min_blob_area,
            dilate_pixels=dilate_pixels,
            box_pad=box_pad
        )

        # save outputs
        np.save(os.path.join(out_dir, f"tile_{i:03d}_prob.npy"), prob.astype(np.float32))
        np.save(os.path.join(out_dir, f"tile_{i:03d}_roi.npy"), roi_mask.astype(np.uint8))
        with open(os.path.join(out_dir, f"tile_{i:03d}_boxes.json"), "w") as f:
            json.dump(boxes, f, indent=2)

        print(f"tile {i:03d}: boxes={len(boxes)} roi_pixels={int(roi_mask.sum())}")

    print("Done:", out_dir)


if __name__ == "__main__":
    main()
