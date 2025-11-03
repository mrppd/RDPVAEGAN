# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:54:38 2023

@author: pronaya
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Utils
# -------
def read_cardio_csv(path):
    try:
        df = pd.read_csv(path, sep=';')
    except Exception:
        df = pd.read_csv(path)
    return df

def standardize_cardio_schema(df):
    df = df.copy()

    # If age seems to be in days, convert to years
    if 'age' in df.columns and df['age'].median() > 120:
        df['age'] = df['age'] / 365.25

    # Normalize height to centimeters 
    if 'height' in df.columns:
        h = df['height'].astype(float)
        if h.median() < 3:
            df['height'] = h * 100.0

    # engineered features used in the training pipeline
    if all(c in df.columns for c in ['weight', 'height']):
        df['bmi'] = df['weight'] / ((df['height'] / 100.0) ** 2 + 1e-9)
    if all(c in df.columns for c in ['ap_hi', 'ap_lo']):
        df['pp'] = df['ap_hi'] - df['ap_lo']

    return df

def split_real(df, label_col, test_size=0.2, seed=17):
    # Keep label for supervised; remove nothing yet for unsupervised
    if label_col in df.columns:
        y = df[label_col].values
        X = df.drop(columns=[label_col])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_test
    else:
        # no label present
        X_train, X_test = train_test_split(df, test_size=test_size, random_state=seed)
        return X_train.reset_index(drop=True), X_test.reset_index(drop=True), None, None

def align_columns(real_df_like, df_to_align):
    """Align df_to_align to real_df_like columns order; drop extras; add missing as zeros."""
    target_cols = list(real_df_like.columns)
    aligned = df_to_align.copy()
    # drop extras
    aligned = aligned[[c for c in aligned.columns if c in target_cols]]
    # add missing
    for c in target_cols:
        if c not in aligned.columns:
            aligned[c] = 0.0
    aligned = aligned[target_cols]
    return aligned


# MMD (RBF kernel)
# -----------------
def _pairwise_sq_dists(X, Y):
    # X: [n,d], Y: [m,d]
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    X_norm = (X**2).sum(1).reshape(-1,1)
    Y_norm = (Y**2).sum(1).reshape(1,-1)
    K = X_norm + Y_norm - 2 * X @ Y.T
    np.maximum(K, 0, out=K)
    return K

def _rbf_kernel(X, Y, sigma):
    d2 = _pairwise_sq_dists(X, Y)
    return np.exp(-d2 / (2.0 * sigma * sigma))

def _median_heuristic_sigma(Z):
    # Z: [n, d]
    idx = np.random.choice(len(Z), size=min(2000, len(Z)), replace=False)
    Zs = Z[idx]
    d2 = _pairwise_sq_dists(Zs, Zs)
    # take median of upper triangle excluding diagonal
    tri = d2[np.triu_indices_from(d2, k=1)]
    med = np.median(tri)
    if med <= 0 or not np.isfinite(med):
        med = 1.0
    return np.sqrt(0.5 * med)

def mmd_rbf(X, Y, sigma=None):
    """
    Unbiased MMD^2 with RBF kernel.
    """
    n = X.shape[0]
    m = Y.shape[0]
    assert n > 1 and m > 1
    if sigma is None:
        sigma = _median_heuristic_sigma(np.vstack([X, Y]))
    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)
    # Unbiased estimators: remove diagonals
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_xx = Kxx.sum() / (n * (n - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean()
    return term_xx + term_yy - 2.0 * term_xy


# Dimension-wise prediction (RF, F1)
# ---------------------------------------
def binarize_series_for_f1(s, strategy="median"):
    s = pd.to_numeric(s, errors="coerce")
    if strategy == "median":
        thr = np.nanmedian(s.values)
        # if all equal or NaN median, fallback 0/1 evenly
        if not np.isfinite(thr):
            thr = np.nanmean(s.values)
        return (s.values > thr).astype(int)
    raise ValueError("Unknown binarization strategy")

def is_onehot_column(colname, all_cols):
    """
    Heuristic: if there are other columns starting with 'name_' or 'name=' namespace.
    """
    base = None
    if '_' in colname:
        base = colname.split('_')[0]
    elif '=' in colname:
        base = colname.split('=')[0]
    if base is None:
        return False
    prefix = base + '_'
    return any((c != colname) and c.startswith(prefix) for c in all_cols)

def dimension_wise_prediction_f1(real_train, real_test, synth, macro=True, seed=17):
    """
    For each column k:
      - y_train := synth[k] (binarized if numeric w/o clear one-hot group)
      - X_train := synth.drop(k)
      - Evaluate on real_test: y_test := real_test[k] (binarized, as above), X_test := real_test.drop(k)
    Returns dict with per-dimension F1 and macro average.
    """
    rng = np.random.RandomState(seed)
    cols = list(real_train.columns)
    f1s = {}

    for k in tqdm(cols, desc="Dimension-wise prediction"):
        # Build train/test matrices
        Xtr = synth.drop(columns=[k])
        Xte = real_test.drop(columns=[k])

        # Determine if binary already (e.g., one-hot)
        # If column looks like one-hot, use 0.5 threshold; else binarize by median
        ytr_raw = synth[k]
        yte_raw = real_test[k]

        if set(np.unique(ytr_raw.round(4))) <= {0.0, 1.0} or is_onehot_column(k, cols):
            ytr = (ytr_raw.values > 0.5).astype(int)
            yte = (yte_raw.values > 0.5).astype(int)
        else:
            ytr = binarize_series_for_f1(ytr_raw)
            yte = binarize_series_for_f1(yte_raw)

        # Train RF on synthetic; evaluate on real test
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=rng, n_jobs=-1, class_weight="balanced"
        )
        clf.fit(Xtr.values, ytr)
        yhat = clf.predict(Xte.values)
        f1 = f1_score(yte, yhat, zero_division=0)
        f1s[k] = f1

    macro_f1 = float(np.mean(list(f1s.values()))) if macro else None
    return f1s, macro_f1


# Supervised eval (AUROC/AUPRC)
# ------------------------------
def supervised_eval(real_train, real_test, y_train, y_test, synth, label_col):
    if label_col not in synth.columns:
        return None

    # Split synth into X/y
    y_s = synth[label_col].values
    X_s = synth.drop(columns=[label_col])

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, random_state=17, n_jobs=-1, class_weight="balanced_subsample"
    )
    clf.fit(X_s.values, y_s)

    # Evaluate on real test
    proba = clf.predict_proba(real_test.values)[:, 1]
    auroc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)
    return {"AUROC": float(auroc), "AUPRC": float(auprc)}


# Main
# ------
def main(args):
    # Load real 
    real_df_raw = read_cardio_csv(args.real_csv)
    real_df_raw = standardize_cardio_schema(real_df_raw)

    # Keep a snapshot of original columns for labeling
    label_col = args.label if args.label in real_df_raw.columns else None

    # One-hot encode categoricals like in training
    # If you used a fixed preprocess before training, ensure the same columns/order here.
    # Detect common categoricals and one-hot them.
    categorical = [c for c in ['gender','cholesterol','gluc','smoke','alco','active'] if c in real_df_raw.columns]
    numeric = [c for c in ['age','height','weight','ap_hi','ap_lo','bmi','pp'] if c in real_df_raw.columns]

    # Build feature frame (exclude label for feature space)
    features = numeric + categorical
    keep_cols = [c for c in features if c in real_df_raw.columns]
    feat_df = real_df_raw[keep_cols].copy()

    # One-hot
    if categorical:
        feat_df = pd.get_dummies(feat_df, columns=categorical, drop_first=False)

    # Z-score numeric (fit on train portion later to avoid leakage)
    # First, attach label if exists so we can split consistently
    if label_col is not None and label_col in real_df_raw.columns:
        feat_df[label_col] = real_df_raw[label_col].astype(int)

    # Split real into train/test BEFORE scaling (fit scaler on train)
    Xtr_real, Xte_real, ytr_real, yte_real = split_real(feat_df, label_col or "__none__", test_size=0.2, seed=17)

    # Standardize numeric columns using train stats, apply to both splits
    num_in_space = [c for c in numeric if c in Xtr_real.columns]
    for c in num_in_space:
        m = Xtr_real[c].mean()
        s = Xtr_real[c].std() + 1e-6
        Xtr_real[c] = (Xtr_real[c] - m) / s
        Xte_real[c] = (Xte_real[c] - m) / s

    # Load synthetic 
    synth_df = pd.read_csv(args.synthetic_csv)
    # If the synthetic file was saved without label, this will simply be absent
    # Align synthetic to the REAL feature space (order & missing cols)
    synth_aligned = align_columns(Xtr_real, synth_df)

    # MMD (use 800 samples each, as in the paper)
    n_mmd = min(800, len(Xtr_real), len(synth_aligned))
    real_mmd_sample = Xtr_real.sample(n=n_mmd, random_state=17).values.astype(np.float64)
    synth_mmd_sample = synth_aligned.sample(n=n_mmd, random_state=42).values.astype(np.float64)
    mmd_score = mmd_rbf(real_mmd_sample, synth_mmd_sample)

    # Dimension-wise prediction (F1) 
    f1_per_dim, macro_f1 = dimension_wise_prediction_f1(
        real_train=Xtr_real, real_test=Xte_real, synth=synth_aligned, macro=True
    )

    # Supervised (AUROC/AUPRC) if labels exist in synthetic
    sup = None
    if args.label and args.label in synth_df.columns and (ytr_real is not None):
        # Make sure real test feature order equals synthetic training order
        Xte_real_supervised = align_columns(synth_df.drop(columns=[args.label]), Xte_real)
        sup = supervised_eval(
            real_train=Xtr_real, real_test=Xte_real_supervised,
            y_train=ytr_real, y_test=yte_real,
            synth=synth_df, label_col=args.label
        )

    # Report
    print("\n================ Evaluation (Cardio) ================")
    print(f"MMD (RBF, unbiased, n={n_mmd} per side): {mmd_score:.4f}")
    print(f"Dimension-wise prediction F1 (macro over features): {macro_f1:.4f}")
    print(f"Top-10 per-dimension F1:\n  " +
          ", ".join(f"{k}:{v:.2f}" for k,v in sorted(f1_per_dim.items(), key=lambda x: -x[1])[:10]))
    if sup is None:
        print("Supervised (AUROC/AUPRC): skipped (synthetic labels not found).")
    else:
        print(f"Supervised (train on synthetic, test on real) -> AUROC: {sup['AUROC']:.3f} | AUPRC: {sup['AUPRC']:.3f}")

    print("\nNotes:")
    print(" - MMD uses the median heuristic for kernel bandwidth and unbiased estimator (as in common GAN eval practice).")
    print(" - Dimension-wise prediction: RF classifier trained on synthetic X\\k->k and evaluated on real test. "
          "Numeric targets are binarized by median for F1.")
    print(" - Supervised metrics: train on synthetic, evaluate on real; if the synthetic CSV includes the "
          "target column, they’re computed automatically.")
    print(" - For δ you keep in training/accounting; this script only evaluates data, not privacy. ")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real_csv", type=str, required=True, help="Path to Kaggle cardio CSV")
    p.add_argument("--synthetic_csv", type=str, required=True, help="Path to synthetic CSV produced by the model")
    p.add_argument("--label", type=str, default="cardio", help="Target column name (if present in synthetic)")
    args = p.parse_args()
    main(args)
