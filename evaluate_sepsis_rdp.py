# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 02:34:42 2023

@author: pronaya
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


# Utils
# ------
def find_psv_files(base_dir):
    pats = [
        os.path.join(base_dir, "*.psv"),
        os.path.join(base_dir, "*/*.psv"),
        os.path.join(base_dir, "*/*/*.psv"),
    ]
    files = []
    for p in pats: files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError(f"No .psv files under {base_dir}")
    return sorted(files)

def infer_numeric_cols(sample_path, drop_labels=("SepsisLabel",)):
    df = pd.read_csv(sample_path, sep="|")
    cols = [c for c in df.columns if c not in drop_labels]
    numeric = []
    for c in cols:
        try:
            pd.to_numeric(df[c].head(50), errors="raise")
            numeric.append(c)
        except Exception:
            pass
    return numeric

def preprocess_episode(df, cols, max_len):
    # keep requested columns; head/pad to max_len; ffill inside episode
    d = df[cols].copy()
    d = d.head(max_len)
    if len(d) < max_len:
        pad = pd.DataFrame({c: [np.nan]*(max_len - len(d)) for c in cols})
        d = pd.concat([d, pad], axis=0, ignore_index=True)
    d = d.ffill()
    return d.values.astype(np.float32)  # (T, D)

def load_real_tensor_flat(real_dir, max_len=48, label_col="SepsisLabel"):
    files = find_psv_files(real_dir)
    cols = infer_numeric_cols(files[0], drop_labels=(label_col,))
    X_list, labels = [], []

    for fp in tqdm(files, desc="Reading .psv"):
        df = pd.read_csv(fp, sep="|")
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        arr = preprocess_episode(df, cols, max_len)  # (T, D)
        X_list.append(arr)

        # derive patient-level label if present per-timepoint
        if label_col in df.columns:
            y = int((df[label_col].fillna(0).astype(float) > 0.5).any())
        else:
            y = None
        labels.append(y)

    X = np.stack(X_list, axis=0)  # (N, T, D)
    # fill NaNs by dataset-wide medians
    flat = X.reshape(-1, X.shape[-1])
    meds = np.nanmedian(flat, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(meds, inds[2])

    # z-score per feature (dataset-wide)
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    Xz = (X - mean) / std

    # flatten to (N, T*D)
    X_flat = Xz.reshape(Xz.shape[0], -1)
    return X_flat.astype(np.float32), cols, np.array(labels, dtype=float)


# Helpers for aligning synthetic columns
# ----------------------------------------
def expected_flattened_columns(feature_names, T):
    cols = []
    for t in range(T):
        for f in feature_names:
            cols.append(f"{f}@t{t}")
    return cols

def align_synth_to_real(synth_df, real_cols):
    # drop extras; add missing (zeros); reorder
    s = synth_df.copy()
    s = s[[c for c in s.columns if c in real_cols or c == "__LABEL__" or c == "SepsisLabel"]]
    for c in real_cols:
        if c not in s.columns:
            s[c] = 0.0
    s = s[real_cols + [c for c in s.columns if c not in real_cols]]  # keep labels (if present) at end
    return s


# MMD (RBF, unbiased; 800/side )
# ---------------------------------
def _pairwise_sq_dists(X, Y):
    Xn = (X**2).sum(1)[:, None]
    Yn = (Y**2).sum(1)[None, :]
    K = Xn + Yn - 2 * X @ Y.T
    np.maximum(K, 0, out=K)
    return K

def _median_heuristic_sigma(Z):
    m = min(2000, len(Z))
    idx = np.random.choice(len(Z), size=m, replace=False)
    Zs = Z[idx]
    d2 = _pairwise_sq_dists(Zs, Zs)
    tri = d2[np.triu_indices_from(d2, k=1)]
    med = np.median(tri)
    if not np.isfinite(med) or med <= 0: med = 1.0
    return np.sqrt(0.5 * med)

def _rbf(X, Y, sigma):
    return np.exp(-_pairwise_sq_dists(X, Y) / (2.0 * sigma * sigma))

def mmd_rbf_unbiased(X, Y, sigma=None):
    n, m = X.shape[0], Y.shape[0]
    assert n > 1 and m > 1
    if sigma is None:
        sigma = _median_heuristic_sigma(np.vstack([X, Y]))
    Kxx, Kyy, Kxy = _rbf(X, X, sigma), _rbf(Y, Y, sigma), _rbf(X, Y, sigma)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_xx = Kxx.sum() / (n * (n - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean()
    return term_xx + term_yy - 2.0 * term_xy


# Dimension-wise prediction (macro-F1)
# --------------------------------------
def is_binary_like(arr, tol=1e-6):
    uniq = np.unique(np.round(arr, 4))
    return set(uniq).issubset({0.0, 1.0})

def binarize_numeric(arr):
    thr = np.nanmedian(arr)
    if not np.isfinite(thr): thr = np.nanmean(arr)
    return (arr > thr).astype(int)

def dimwise_macro_f1(real_train_df, real_test_df, synth_df, seed=17):
    cols = list(real_train_df.columns)
    f1s = {}

    for k in tqdm(cols, desc="Dimension-wise prediction"):
        Xtr = synth_df.drop(columns=[k]).values
        Xte = real_test_df.drop(columns=[k]).values

        ytr_raw = synth_df[k].values
        yte_raw = real_test_df[k].values

        if is_binary_like(ytr_raw):
            ytr = (ytr_raw > 0.5).astype(int)
            yte = (yte_raw > 0.5).astype(int)
        else:
            ytr = binarize_numeric(ytr_raw)
            yte = binarize_numeric(yte_raw)

        clf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed, class_weight="balanced"
        )
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        f1s[k] = f1_score(yte, yhat, zero_division=0)

    macro_f1 = float(np.mean(list(f1s.values())))
    return f1s, macro_f1


# Supervised: AUROC/AUPRC
# ------------------------
def supervised_metrics(synth_with_label, real_test_with_label, label_col):
    if label_col not in synth_with_label.columns: return None

    y_s = synth_with_label[label_col].values.astype(int)
    X_s = synth_with_label.drop(columns=[label_col]).values

    y_te = real_test_with_label[label_col].values.astype(int)
    X_te = real_test_with_label.drop(columns=[label_col]).values

    clf = RandomForestClassifier(
        n_estimators=400, n_jobs=-1, random_state=17, class_weight="balanced_subsample"
    )
    clf.fit(X_s, y_s)
    proba = clf.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_te, proba)
    auprc = average_precision_score(y_te, proba)
    return {"AUROC": float(auroc), "AUPRC": float(auprc)}


# Main
# ------
def main(args):
    # Load REAL, flatten to (N, T*D) 
    X_flat, feat_names, patient_labels = load_real_tensor_flat(
        args.real_dir, max_len=args.max_len, label_col=args.derive_label_from
    )

    # Build flattened column names to match training/synthetic convention
    flattened_cols = expected_flattened_columns(feat_names, args.max_len)
    real_df = pd.DataFrame(X_flat, columns=flattened_cols)

    # Attach patient-level label if derived
    if args.derive_label_from is not None and np.isfinite(patient_labels).any():
        real_df[args.derive_label_from] = patient_labels

    # Split real into train/test BEFORE any alignment
    label_for_split = args.label_col if (args.label_col and args.label_col in real_df.columns) else "__no_label__"
    if label_for_split in real_df.columns:
        y = real_df[label_for_split].values.astype(int)
        X = real_df.drop(columns=[label_for_split])
        Xtr_real, Xte_real, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
        # reattach for supervised eval later
        Xtr_real[label_for_split] = ytr
        Xte_real[label_for_split] = yte
    else:
        Xtr_real, Xte_real = train_test_split(real_df, test_size=0.2, random_state=17)
        ytr = yte = None

    # Load SYNTH; align columns to REAL feature space
    synth_df = pd.read_csv(args.synthetic_csv)
    synth_aligned = align_synth_to_real(synth_df, list(Xtr_real.drop(columns=[c for c in [label_for_split] if c in Xtr_real.columns]).columns))

    # MMD (800 per side) over the feature space (no labels) 
    n_mmd = min(800, len(Xtr_real), len(synth_aligned))
    X_mmd_real = Xtr_real.drop(columns=[c for c in [label_for_split] if c in Xtr_real.columns]).sample(n=n_mmd, random_state=17).values.astype(np.float64)
    X_mmd_synth = synth_aligned.sample(n=n_mmd, random_state=42).values.astype(np.float64)
    mmd = mmd_rbf_unbiased(X_mmd_real, X_mmd_synth)

    # Dimension-wise prediction (macro-F1)
    f1_per_dim, macro_f1 = dimwise_macro_f1(
        real_train_df=Xtr_real.drop(columns=[c for c in [label_for_split] if c in Xtr_real.columns]),
        real_test_df=Xte_real.drop(columns=[c for c in [label_for_split] if c in Xte_real.columns]),
        synth_df=synth_aligned
    )

    # Supervised AUROC/AUPRC 
    sup = None
    if args.label_col and args.label_col in synth_df.columns and label_for_split in Xte_real.columns:
        # Align both sides to same order incl. label at the end
        synth_sup = synth_df.copy()
        if args.label_col not in synth_sup.columns:
            pass  # skip if not there
        else:
            ordered = list(Xtr_real.drop(columns=[label_for_split]).columns)
            synth_sup = align_synth_to_real(synth_sup, ordered)
            synth_sup[args.label_col] = synth_df[args.label_col].values
            real_te_sup = Xte_real.rename(columns={label_for_split: args.label_col})
            sup = supervised_metrics(synth_sup, real_te_sup, args.label_col)

    # Report
    print("\n================ Evaluation (Sepsis) ================")
    print(f"MMD (RBF, unbiased, n={n_mmd} per side): {mmd:.4f}")
    print(f"Dimension-wise prediction F1 (macro over {len(f1_per_dim)} columns): {macro_f1:.4f}")
    top10 = ", ".join(f"{k}:{v:.2f}" for k,v in sorted(f1_per_dim.items(), key=lambda x: -x[1])[:10])
    print(f"Top-10 per-dimension F1: {top10 if top10 else 'n/a'}")
    if sup is None:
        print("Supervised (AUROC/AUPRC): skipped (no labels in synthetic).")
    else:
        print(f"Supervised (train on synthetic, test on real) -> AUROC: {sup['AUROC']:.3f} | AUPRC: {sup['AUPRC']:.3f}")

    print("\nNotes:")
    print(" - Real episodes are standardized and flattened to match the training space (T*D) and the synthetic CSV.")
    print(" - MMD uses the median heuristic bandwidth and the unbiased estimator, comparing 800 samples per side .")
    print(" - Dimension-wise prediction: for each column k, RF is trained on synthetic X\\k->k and evaluated on real.")
    print(" - If the synthetic includes a patient-level label, AUROC/AUPRC are also computed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=str, required=True, help="Folder with Sepsis .psv files")
    ap.add_argument("--synthetic_csv", type=str, required=True, help="Synthetic CSV produced by the model")
    ap.add_argument("--max_len", type=int, default=48, help="Fixed horizon used in training/preprocess")
    ap.add_argument("--label_col", type=str, default=None, help="Synthetic label column name if present (e.g., SepsisLabel)")
    ap.add_argument("--derive_label_from", type=str, default="SepsisLabel", help="Per-timepoint label in real .psv used to derive patient label; set None to disable")
    args = ap.parse_args()
    main(args)
