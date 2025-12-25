# # stage1_train.py
# import os
# import json
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import roc_auc_score, average_precision_score


# def build_pixel_dataset(X_tiles, y_tiles, neg_per_pos=20, seed=0):
#     """
#     X_tiles: (N, H, W, C)
#     y_tiles: (N, H, W) 0/1
#     Returns:
#       X: (N_samples, C)
#       y: (N_samples,)
#     Strategy:
#       - take ALL positives
#       - sample negatives at ratio neg_per_pos : 1
#     """
#     rng = np.random.default_rng(seed)

#     N, H, W, C = X_tiles.shape
#     X_flat = X_tiles.reshape(-1, C)
#     y_flat = y_tiles.reshape(-1)

#     pos_idx = np.where(y_flat == 1)[0]
#     neg_idx = np.where(y_flat == 0)[0]

#     n_pos = pos_idx.size
#     n_neg_take = min(neg_idx.size, n_pos * neg_per_pos)

#     neg_take = rng.choice(neg_idx, size=n_neg_take, replace=False)

#     idx = np.concatenate([pos_idx, neg_take])
#     rng.shuffle(idx)

#     X = X_flat[idx]
#     y = y_flat[idx].astype(np.uint8)

#     return X, y, {"n_pos": int(n_pos), "n_neg_take": int(n_neg_take), "neg_per_pos": neg_per_pos}


# def main():
#     # Update paths if yours differ
#     train_x_path = "data/np_data/train_x.npy"
#     train_y_path = "data/np_data/train_y.npy"
#     test_x_path  = "data/np_data/test_x.npy"
#     test_y_path  = "data/np_data/test_y.npy"

#     out_dir = "stage1_model"
#     os.makedirs(out_dir, exist_ok=True)

#     print("Loading npy...")
#     X_train_tiles = np.load(train_x_path)
#     y_train_tiles = np.load(train_y_path)
#     X_test_tiles  = np.load(test_x_path)
#     y_test_tiles  = np.load(test_y_path)

#     # ---- Build sampled pixel dataset ----
#     # Start with 20 negatives per positive. If too many false positives later, increase to 50.
#     Xtr, ytr, info = build_pixel_dataset(X_train_tiles, y_train_tiles, neg_per_pos=75, seed=0)
#     Xte, yte, _    = build_pixel_dataset(X_test_tiles,  y_test_tiles,  neg_per_pos=20, seed=1)

#     print("Pixel dataset info:", info)
#     print("Train samples:", Xtr.shape, "pos rate:", ytr.mean())
#     print("Test  samples:", Xte.shape, "pos rate:", yte.mean())

#     # ---- Model: StandardScaler + Logistic Regression ----
#     # class_weight balanced helps a lot with this imbalance
#     model = Pipeline([
#         ("scaler", StandardScaler(with_mean=True, with_std=True)),
#         ("clf", LogisticRegression(
#             max_iter=2000,
#             solver="saga",
#             n_jobs=-1,
#             class_weight="balanced",
#             penalty="l2",
#             C=1.0
#         ))
#     ])

#     print("Training...")
#     model.fit(Xtr, ytr)

#     # ---- Quick evaluation on sampled pixels (not final ROI metric, but sanity) ----
#     print("Evaluating on sampled pixels...")
#     p = model.predict_proba(Xte)[:, 1]
#     roc = roc_auc_score(yte, p)
#     ap = average_precision_score(yte, p)
#     print("Sampled pixel ROC-AUC:", roc)
#     print("Sampled pixel PR-AUC :", ap)

#     # Save model as joblib
#     import joblib
#     joblib.dump(model, os.path.join(out_dir, "logreg.joblib"))

#     # Save metadata
#     meta = {
#         "model": "StandardScaler + LogisticRegression",
#         "neg_per_pos": info["neg_per_pos"],
#         "n_pos": info["n_pos"],
#         "n_neg_take": info["n_neg_take"],
#         "note": "Trained on sampled pixels: all positives + neg_per_pos*positives negatives. Split was by tile via provided train/test npy."
#     }
#     with open(os.path.join(out_dir, "meta.json"), "w") as f:
#         json.dump(meta, f, indent=2)

#     print("Saved:", out_dir)


# if __name__ == "__main__":
#     main()


'''
This is the streaming version to avoid fitting large chunk of data in memory at once.
'''
import os, json
import numpy as np
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


def sample_pixels_from_tile(tile_x, tile_y, neg_per_pos=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    H, W, C = tile_x.shape
    X = tile_x.reshape(-1, C)
    y = tile_y.reshape(-1)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = pos_idx.size

    if n_pos == 0:
        n_neg_take = min(5000, neg_idx.size)
        idx = rng.choice(neg_idx, size=n_neg_take, replace=False)
    else:
        n_neg_take = min(neg_idx.size, n_pos * neg_per_pos)
        neg_take = rng.choice(neg_idx, size=n_neg_take, replace=False)
        idx = np.concatenate([pos_idx, neg_take])
        rng.shuffle(idx)

    return X[idx].astype(np.float32, copy=False), y[idx].astype(np.uint8, copy=False)


def iter_train_batches(X_tiles, y_tiles, neg_per_pos, batch_cap, seed):
    rng = np.random.default_rng(seed)
    N = X_tiles.shape[0]
    for i in range(N):
        X_s, y_s = sample_pixels_from_tile(X_tiles[i], y_tiles[i], neg_per_pos=neg_per_pos, rng=rng)
        n = X_s.shape[0]
        for start in range(0, n, batch_cap):
            end = min(n, start + batch_cap)
            yield X_s[start:end], y_s[start:end]


def main():
    train_x_path = "data/np_data/train_x.npy"
    train_y_path = "data/np_data/train_y.npy"
    test_x_path  = "data/np_data/test_x.npy"
    test_y_path  = "data/np_data/test_y.npy"

    out_dir = "stage1_model_streaming"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading tiles (mmap)...")
    X_train = np.load(train_x_path, mmap_mode="r")
    y_train = np.load(train_y_path, mmap_mode="r")
    X_test  = np.load(test_x_path,  mmap_mode="r")
    y_test  = np.load(test_y_path,  mmap_mode="r")

    # ---- knobs ----
    neg_per_pos = 50
    batch_cap = 100_000     # lower if RAM tight
    epochs = 2
    seed = 0

    # ---- 1) FIRST PASS: fit scaler only (streaming) ----
    scaler = StandardScaler(with_mean=True, with_std=True)
    print("\nPass 1: fitting scaler...")
    b = 0
    for Xb, yb in iter_train_batches(X_train, y_train, neg_per_pos, batch_cap, seed):
        scaler.partial_fit(Xb)
        b += 1
        if b % 20 == 0:
            print(f"  scaler batches: {b}")

    # ---- 2) SECOND PASS: train classifier with frozen scaler ----
    # IMPORTANT: weights should match the sampled distribution, not the full dataset.
    # With neg_per_pos=50, a simple stable choice is:
    class_weight = {0: 1.0, 1: float(neg_per_pos)}
    print("\nUsing class_weight:", class_weight)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        max_iter=1,     # we control epochs
        tol=None,
        class_weight=class_weight,
        average=True,   # stabilizes online SGD a lot
        random_state=seed
    )
    classes = np.array([0, 1], dtype=np.uint8)

    print("\nPass 2: training classifier...")
    first = True
    for ep in range(epochs):
        print(f"\n=== epoch {ep+1}/{epochs} ===")
        batch_num = 0
        for Xb, yb in iter_train_batches(X_train, y_train, neg_per_pos, batch_cap, seed + ep):
            Xb_s = scaler.transform(Xb)
            if first:
                clf.partial_fit(Xb_s, yb, classes=classes)
                first = False
            else:
                clf.partial_fit(Xb_s, yb)

            batch_num += 1
            if batch_num % 20 == 0:
                print(f"  train batches: {batch_num}")

    # ---- quick eval on sampled pixels ----
    print("\nSampling test pixels for quick eval...")
    K = min(10, X_test.shape[0])
    rng = np.random.default_rng(123)

    X_eval_list, y_eval_list = [], []
    for i in range(K):
        Xs, ys = sample_pixels_from_tile(X_test[i], y_test[i], neg_per_pos=neg_per_pos, rng=rng)
        if Xs.shape[0] > 150_000:
            idx = rng.choice(Xs.shape[0], size=150_000, replace=False)
            Xs, ys = Xs[idx], ys[idx]
        X_eval_list.append(Xs)
        y_eval_list.append(ys)

    X_eval = np.concatenate(X_eval_list, axis=0)
    y_eval = np.concatenate(y_eval_list, axis=0)

    X_eval_s = scaler.transform(X_eval)
    p = clf.predict_proba(X_eval_s)[:, 1]
    roc = roc_auc_score(y_eval, p)
    ap = average_precision_score(y_eval, p)
    print("Sampled pixel ROC-AUC:", roc)
    print("Sampled pixel PR-AUC :", ap)

    joblib.dump({"scaler": scaler, "clf": clf}, os.path.join(out_dir, "sgd_logreg.joblib"))
    meta = dict(
        model="Two-pass streaming: StandardScaler.partial_fit then SGDClassifier(log_loss)",
        neg_per_pos=neg_per_pos,
        batch_cap=batch_cap,
        epochs=epochs,
        class_weight=class_weight,
        note="Scaler frozen during classifier training; weights match sampled ratio.",
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
