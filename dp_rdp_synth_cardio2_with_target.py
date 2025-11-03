# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 02:41:52 2023

@author: prona
"""

import os
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from opacus import PrivacyEngine


# Config
# ========
@dataclass
class Config:
    # Kaggle CSV
    data_path: str = "./cardio_train.csv"
    out_csv: str = "synthetic_cardio.csv"

    batch_size: int = 256           # smaller for DP stability
    seed: int = 17
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # VAE
    latent_dim: int = 32
    vae_hidden: Tuple[int, ...] = (128, 128)
    vae_epochs: int = 20
    vae_lr: float = 1e-4            # reduced LR for stability
    vae_weight_decay: float = 1e-4  # helps keep decoder outputs bounded
    vae_target_epsilon: float = 3.0

    # Output bounding for decoder: tanh * out_scale
    out_scale: float = 5.0

    # KL warmup
    vae_beta_max: float = 0.1
    kl_warmup_epochs: int = 10

    # GAN (frozen VAE decoder)
    noise_dim: int = 64
    gen_hidden: Tuple[int, ...] = (128, 128)
    disc_hidden: Tuple[int, ...] = (256, 128)
    gan_epochs: int = 25
    d_lr: float = 1e-3
    g_lr: float = 1e-3
    d_target_epsilon: float = 4.0

    # DP Labeler (for synthetic 'cardio')
    labeler_hidden: Tuple[int, ...] = (64,)
    labeler_epochs: int = 8
    labeler_lr: float = 1e-3
    labeler_target_epsilon: float = 2.0

    # Privacy
    delta: float = 1e-5
    max_grad_norm: float = 0.5      # a bit tighter clipping
    secure_mode: bool = False       # set True if torchcsprng installed

    # Synthesis
    n_synth: int = 50000  # number of synthetic rows to output

    # Utility (TSTR) 
    tstr_train_max: int = 50000     # how many synthetic rows to train on (<= n_synth)
    tstr_test_max: int = 60000      # how many real rows to test on

cfg = Config()

def set_all_seeds(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(cfg.seed)


# Data loading & preprocessing (tabular)
# =======================================
NUMERICS = ["age", "height", "weight", "ap_hi", "ap_lo"]
CATEGORICAL = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
TARGET = "cardio"

def load_cardio_tabular(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at {path}. "
        )
    # Most versions use ';' delimiter
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path)
    return df

def preprocess(df: pd.DataFrame):
    """
    - Drops identifier columns (id/ID/Id)
    - Converts age in days -> years (if needed)
    - Ensures height in cm
    - Adds BMI and pulse pressure
    - One-hot encodes categoricals
    - Z-scores numeric columns
    Returns standardized feature matrix (label excluded) + meta
    """
    df = df.copy()

    # Age (days -> years) if values look large
    if df["age"].median() > 120:
        df["age"] = (df["age"] / 365.25).astype(float)

    # Height: if already meters (<3), convert to cm
    h = df["height"].astype(float)
    if h.median() < 3:
        df["height"] = h * 100.0

    # Feature engineering
    df["bmi"] = df["weight"] / ((df["height"] / 100.0) ** 2 + 1e-9)
    df["pp"] = df["ap_hi"] - df["ap_lo"]

    # Remove label from features if present
    feat_df = df.drop(columns=[TARGET]) if TARGET in df.columns else df

    # Drop identifier columns (huge scale, non-predictive)
    for col in ("id", "ID", "Id"):
        if col in feat_df.columns:
            feat_df = feat_df.drop(columns=[col])

    numeric_cols = [c for c in (NUMERICS + ["bmi", "pp"]) if c in feat_df.columns]
    cat_cols = [c for c in CATEGORICAL if c in feat_df.columns]

    # Soft-clip numeric outliers
    def soft_clip(s: pd.Series, q=0.005):
        lo, hi = s.quantile(q), s.quantile(1 - q)
        return s.clip(lo, hi)

    for c in numeric_cols:
        feat_df[c] = soft_clip(feat_df[c].astype(float))

    # One-hot encode categoricals
    feat_df[cat_cols] = feat_df[cat_cols].astype(int)
    df_oh = pd.get_dummies(feat_df, columns=cat_cols, drop_first=False)

    # Z-score numeric columns
    stats = {}
    for c in numeric_cols:
        m, s = df_oh[c].mean(), df_oh[c].std() + 1e-6
        stats[c] = (m, s)
        df_oh[c] = (df_oh[c] - m) / s

    feature_names = list(df_oh.columns)
    meta = {"numeric_stats": stats, "oh_cols": feature_names, "numeric_cols": numeric_cols}
    return df_oh, feature_names, meta

class CardioDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]


# Models (tabular VAE + GAN) with stability tweaks
# ==================================================
class TanhScaled(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return torch.tanh(x) * self.scale

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: Tuple[int, ...], out_scale: float):
        super().__init__()
        h = list(hidden)
        enc_layers, last = [], input_dim
        for w in h:
            enc_layers += [nn.Linear(last, w), nn.ReLU()]
            last = w
        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(last, latent_dim)
        self.logvar = nn.Linear(last, latent_dim)

        dec_layers, last = [], latent_dim
        for w in reversed(h):
            dec_layers += [nn.Linear(last, w), nn.ReLU()]
            last = w
        dec_layers += [nn.Linear(last, input_dim), TanhScaled(out_scale)]  # bounded outputs
        self.decoder = nn.Sequential(*dec_layers)

        # Xavier init for better numeric stability under DP
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        # Clamp log-variance to avoid extreme std under DP noise
        logvar = self.logvar(h).clamp(min=-6.0, max=2.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z)  # already bounded by TanhScaled
        return xhat, mu, logvar

def vae_loss(x, xhat, mu, logvar, beta=0.1):
    # SmoothL1 (Huber) is more robust under DP than MSE for occasional spikes
    recon = F.smooth_l1_loss(xhat, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon.detach(), kld.detach()

class Generator(nn.Module):
    def __init__(self, noise_dim: int, latent_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers, last = [], noise_dim
        for w in hidden:
            layers += [nn.Linear(last, w), nn.ReLU()]
            last = w
        layers += [nn.Linear(last, latent_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers, last = [], input_dim
        for w in hidden:
            layers += [nn.Linear(last, w), nn.LeakyReLU(0.2)]
            last = w
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x).squeeze(-1)

# Simple MLP classifier for DP labeler
class MLPLabeler(nn.Module):
    def __init__(self, input_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers, last = [], input_dim
        for w in hidden:
            layers += [nn.Linear(last, w), nn.ReLU()]
            last = w
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x).squeeze(-1)


# DP helpers (using PrivacyEngine(accountant="rdp"))
# ====================================================
def make_private_with_target_eps(model, optimizer, data_loader,
                                 target_epsilon, target_delta, epochs,
                                 max_grad_norm, secure_mode):
    """
    Wrap model/optimizer/loader with Opacus PrivacyEngine.
    NOTE: We specify accountant="rdp" (string).
    """
    pe = PrivacyEngine(accountant="rdp", secure_mode=secure_mode)
    model, optimizer, private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
    )
    return model, optimizer, private_loader, pe

def unwrap_opacus_module(m: nn.Module) -> nn.Module:
    return getattr(m, "_module", m)

# Build a fresh, unhooked decoder with trained weights from the (wrapped) VAE
@torch.no_grad()
def build_clean_frozen_decoder(vae_wrapped: nn.Module, input_dim: int, cfg: Config) -> nn.Module:
    vae_base = unwrap_opacus_module(vae_wrapped)
    decoder_sd = vae_base.decoder.state_dict()
    # Fresh VAE to host a clean decoder
    vae_fresh = VAE(input_dim, cfg.latent_dim, cfg.vae_hidden, cfg.out_scale)
    vae_fresh.decoder.load_state_dict(decoder_sd)
    decoder = vae_fresh.decoder
    for p in decoder.parameters():
        p.requires_grad = False
    decoder.eval()
    decoder = decoder.to(cfg.device)
    return decoder

def train_vae_dp(model: VAE, train_loader, cfg: Config):
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr, weight_decay=cfg.vae_weight_decay)

    model, opt, priv_loader, pe = make_private_with_target_eps(
        model, opt, train_loader,
        target_epsilon=cfg.vae_target_epsilon,
        target_delta=cfg.delta,
        epochs=cfg.vae_epochs,
        max_grad_norm=cfg.max_grad_norm,
        secure_mode=cfg.secure_mode,
    )

    model.train()
    for e in range(cfg.vae_epochs):
        beta = cfg.vae_beta_max * min(1.0, (e + 1) / max(1, cfg.kl_warmup_epochs))

        losses, maes = [], []
        for x in priv_loader:
            x = x.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            xhat, mu, logvar = model(x)
            loss, _, _ = vae_loss(x, xhat, mu, logvar, beta=beta)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf detected in VAE loss.")
            loss.backward()
            opt.step()
            losses.append(loss.item())
            with torch.no_grad():
                maes.append((xhat - x).abs().mean().item())

        eps = pe.get_epsilon(cfg.delta)
        print(f"[VAE {e+1}/{cfg.vae_epochs}] loss={np.mean(losses):.6f} mae={np.mean(maes):.6f} | "
              f"beta={beta:.4f} | eps_so_far={eps:.3f}, delta={cfg.delta}")
    return model, pe

def _freeze_all_params(m: nn.Module, freeze: bool = True):
    for p in m.parameters():
        p.requires_grad = not freeze

def train_gan_dp(decoder: nn.Module, input_dim: int, train_loader, cfg: Config):
    decoder = decoder.to(cfg.device)
    G = Generator(cfg.noise_dim, cfg.latent_dim, cfg.gen_hidden).to(cfg.device)
    D = Discriminator(input_dim, cfg.disc_hidden).to(cfg.device)

    optD = torch.optim.Adam(D.parameters(), lr=cfg.d_lr, betas=(0.5, 0.999))
    D, optD, priv_loader, peD = make_private_with_target_eps(
        D, optD, train_loader,
        target_epsilon=cfg.d_target_epsilon,
        target_delta=cfg.delta,
        epochs=cfg.gan_epochs,
        max_grad_norm=cfg.max_grad_norm,
        secure_mode=cfg.secure_mode,
    )

    D_shadow = Discriminator(input_dim, cfg.disc_hidden).to(cfg.device)
    _freeze_all_params(D_shadow, freeze=True)
    D_shadow.eval()
    D_base = unwrap_opacus_module(D)
    D_shadow.load_state_dict(D_base.state_dict())

    optG = torch.optim.Adam(G.parameters(), lr=cfg.g_lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    for e in range(cfg.gan_epochs):
        d_losses, g_losses = [], []
        print(f"[DEBUG] GAN epoch {e+1}/{cfg.gan_epochs}")
        for real in priv_loader:
            real = real.to(cfg.device)
            bs = real.size(0)

            # D step (private)
            D.train()
            optD.zero_grad(set_to_none=True)
            z = torch.randn(bs, cfg.noise_dim, device=cfg.device)
            with torch.no_grad():
                fake = decoder(G(z))
            lossD = bce(D(real), torch.ones(bs, device=cfg.device)) + \
                    bce(D(fake), torch.zeros(bs, device=cfg.device))
            if not torch.isfinite(lossD):
                raise RuntimeError("NaN/Inf in D loss.")
            lossD.backward()
            optD.step()
            d_losses.append(lossD.item())

            # Sync shadow
            D_base = unwrap_opacus_module(D)
            D_shadow.load_state_dict(D_base.state_dict())

            # G step (non-private; uses shadow D)
            optG.zero_grad(set_to_none=True)
            z = torch.randn(bs, cfg.noise_dim, device=cfg.device)
            fake_g = decoder(G(z))
            logits = D_shadow(fake_g)
            lossG = bce(logits, torch.ones_like(logits))
            if not torch.isfinite(lossG):
                raise RuntimeError("NaN/Inf in G loss.")
            lossG.backward()
            optG.step()
            g_losses.append(lossG.item())

        eps = peD.get_epsilon(cfg.delta)
        print(f"[GAN {e+1}/{cfg.gan_epochs}] D={np.mean(d_losses):.6f} G={np.mean(g_losses):.6f} | "
              f"eps_so_far={eps:.3f}, delta={cfg.delta}")

    return G, D, peD


# Post-processing: invert z-score & collapse one-hots
# =====================================================
def invert_to_raw_df(Xsyn: np.ndarray, feat_names: list, meta: dict) -> pd.DataFrame:
    df = pd.DataFrame(Xsyn, columns=feat_names).copy()

    # Invert z-score on numeric columns
    for col, (m, s) in meta.get("numeric_stats", {}).items():
        if col in df.columns:
            df[col] = df[col] * s + m

    # Collapse one-hot groups
    cat_bases = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
    for base in cat_bases:
        oh_cols = [c for c in df.columns if c.startswith(base + "_")]
        if not oh_cols:
            continue
        idx = df[oh_cols].values.argmax(axis=1)
        labels = [int(c.split("_", 1)[1]) for c in oh_cols]
        df[base] = [labels[i] for i in idx]
        df.drop(columns=oh_cols, inplace=True)

    # Cast/clip
    for c in ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "ap_hi", "ap_lo"]:
        if c in df.columns:
            df[c] = np.rint(df[c]).astype(int)
    if "gender" in df.columns:
        df["gender"] = df["gender"].clip(1, 2)
    if "cholesterol" in df.columns:
        df["cholesterol"] = df["cholesterol"].clip(1, 3)
    if "gluc" in df.columns:
        df["gluc"] = df["gluc"].clip(1, 3)
    for c in ["smoke", "alco", "active"]:
        if c in df.columns:
            df[c] = df[c].clip(0, 1)

    # Age back to days
    if "age" in df.columns:
        df["age"] = np.rint(df["age"] * 365.25).astype(int)

    # Recompute engineered features from raw numerics
    for c in ["bmi", "pp"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    if {"height", "weight"} <= set(df.columns):
        df["bmi"] = df["weight"] / ((df["height"] / 100.0) ** 2 + 1e-9)
    if {"ap_hi", "ap_lo"} <= set(df.columns):
        df["pp"] = df["ap_hi"] - df["ap_lo"]

    # Rounding
    if "height" in df.columns:
        df["height"] = np.rint(df["height"]).astype(int)
    for c in ["weight", "bmi"]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    order = ["age","gender","height","weight","ap_hi","ap_lo",
             "cholesterol","gluc","smoke","alco","active","bmi","pp"]
    cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    return df[cols]


# DP labeler & utility checks
# ============================
def train_dp_labeler(X: np.ndarray, y: np.ndarray, cfg: Config):
    ds = CardioDataset(X, y)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model = MLPLabeler(X.shape[1], cfg.labeler_hidden).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.labeler_lr)
    bce = nn.BCEWithLogitsLoss()

    model, opt, priv_loader, pe = make_private_with_target_eps(
        model, opt, loader,
        target_epsilon=cfg.labeler_target_epsilon,
        target_delta=cfg.delta,
        epochs=cfg.labeler_epochs,
        max_grad_norm=cfg.max_grad_norm,
        secure_mode=cfg.secure_mode,
    )

    for e in range(cfg.labeler_epochs):
        losses = []
        for xb, yb in priv_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bce(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf in labeler loss")
            loss.backward()
            opt.step()
            losses.append(loss.item())
        eps = pe.get_epsilon(cfg.delta)
        print(f"[Labeler {e+1}/{cfg.labeler_epochs}] loss={np.mean(losses):.4f} | eps_so_far={eps:.3f}")

    return model, pe

@torch.no_grad()
def predict_labels(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    logits = model(X_t)
    probs = torch.sigmoid(logits).cpu().numpy()
    # Sample Bernoulli to get binary labels
    y_syn = (np.random.rand(len(probs)) < probs).astype(int)
    return y_syn, probs

def ks_tests_numeric(real_df: pd.DataFrame, syn_df: pd.DataFrame, meta: dict):
    # Compare standardized numeric columns (z-space), since that's consistent across sets
    try:
        from scipy.stats import ks_2samp
    except Exception:
        print("[WARN] scipy not installed. Run: pip install scipy")
        return []
    cols = [c for c in meta.get("numeric_cols", []) if c in real_df.columns and c in syn_df.columns]
    results = []
    for c in cols:
        stat, p = ks_2samp(real_df[c].values, syn_df[c].values)
        results.append((c, float(stat), float(p)))
    return results

def tstr_auc(X_syn: np.ndarray, y_syn: np.ndarray, X_real: np.ndarray, y_real: np.ndarray, cfg: Config):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except Exception:
        print("[WARN] scikit-learn not installed. Run: pip install scikit-learn")
        return None

    n_train = min(cfg.tstr_train_max, len(X_syn))
    n_test = min(cfg.tstr_test_max, len(X_real))
    idx_train = np.random.choice(len(X_syn), n_train, replace=False)
    idx_test = np.random.choice(len(X_real), n_test, replace=False)

    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    clf.fit(X_syn[idx_train], y_syn[idx_train])
    probs = clf.predict_proba(X_real[idx_test])[:,1]
    auc = roc_auc_score(y_real[idx_test], probs)
    return float(auc)


# Synthesis
# ==========
@torch.no_grad()
def sample_synthetic(G: Generator, decoder: nn.Module, n: int, cfg: Config) -> np.ndarray:
    G.eval(); decoder.eval()
    out = []
    bs = 4096
    for i in range(0, n, bs):
        z = torch.randn(min(bs, n - i), cfg.noise_dim, device=cfg.device)
        x = decoder(G(z))
        out.append(x.detach().cpu().numpy())
    return np.concatenate(out, axis=0)

def save_synth(X: np.ndarray, feature_names: List[str], cfg: Config):
    df = pd.DataFrame(X, columns=feature_names)
    df.to_csv(cfg.out_csv, index=False)
    print(f"Saved synthetic: {cfg.out_csv} shape={df.shape}")


# Main
# =====
def main(cfg: Config):
    # Load & preprocess
    df_raw = load_cardio_tabular(cfg.data_path)
    y_real = df_raw[TARGET].values.astype(np.float32) if TARGET in df_raw.columns else None
    X_df, feat_names, meta = preprocess(df_raw)

    # Update delta ~ 1/N (conventional choice; keep <= provided)
    N = len(X_df)
    cfg.delta = min(cfg.delta, 1.0 / max(N, 10_000))
    print(f"Dataset rows: {N} | Features (after OH & scaling): {len(feat_names)} | delta set to {cfg.delta:.2e}")

    # Torch dataset/loader (Opacus will replace sampler internally)
    X_np = X_df.values.astype(np.float32)

    # Stage A: DP-VAE
    input_dim = X_np.shape[1]
    vae = VAE(input_dim, cfg.latent_dim, cfg.vae_hidden, cfg.out_scale)
    ds = CardioDataset(X_np)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    vae, pe_vae = train_vae_dp(vae, loader, cfg)
    eps_stage_a = pe_vae.get_epsilon(cfg.delta)
    print(f"[Privacy after Stage A] epsilon={eps_stage_a:.3f}, delta={cfg.delta}")
    print("[INFO] Building clean frozen decoder...")

    # Stage B: DP-GAN with shadow discriminator
    decoder = build_clean_frozen_decoder(vae, input_dim, cfg)
    print("[INFO] Decoder ready. Starting DP-GAN training...")
    G, D, pe_disc = train_gan_dp(decoder, input_dim, loader, cfg)
    eps_stage_b = pe_disc.get_epsilon(cfg.delta)
    print(f"[Privacy for Stage B] epsilon={eps_stage_b:.3f}, delta={cfg.delta}")

    # Synthesize 
    Xsyn = sample_synthetic(G, decoder, cfg.n_synth, cfg)
    save_synth(Xsyn, feat_names, cfg)  # standardized space

    # Post-process to dataset-like units/categories
    df_post = invert_to_raw_df(Xsyn, feat_names, meta)
    df_post.to_csv("synthetic_cardio_post.csv", index=False)
    print("Saved postprocessed: synthetic_cardio_post.csv", df_post.shape)

    # DP Labeler (on real standardized features)
    if y_real is not None:
        labeler, pe_lab = train_dp_labeler(X_np, y_real, cfg)
        eps_labeler = pe_lab.get_epsilon(cfg.delta)
        print(f"[Privacy for Labeler] epsilon={eps_labeler:.3f}, delta={cfg.delta}")
    else:
        raise RuntimeError("Real labels not found; cannot train DP labeler.")

    # Predict labels for synthetic (standardized) and save
    y_syn, y_syn_prob = predict_labels(labeler.to(cfg.device), Xsyn, cfg.device)
    df_syn_lab = pd.DataFrame(Xsyn, columns=feat_names)
    df_syn_lab[TARGET] = y_syn
    df_syn_lab.to_csv("synthetic_cardio_labeled.csv", index=False)
    print("Saved model-space labeled: synthetic_cardio_labeled.csv", df_syn_lab.shape)

    # Add labels to postprocessed file too (append same y_syn order)
    df_post_lab = df_post.copy()
    df_post_lab[TARGET] = y_syn
    df_post_lab.to_csv("synthetic_cardio_post_labeled.csv", index=False)
    print("Saved postprocessed labeled: synthetic_cardio_post_labeled.csv", df_post_lab.shape)

    # Utility checks
    print("[Utility] KS tests on standardized numeric columns")
    ks = ks_tests_numeric(X_df, pd.DataFrame(Xsyn, columns=feat_names), meta)
    if ks:
        for c, stat, p in ks:
            print(f"KS {c:>10s}: stat={stat:.3f}, p={p:.3g}")

    print("[Utility] TSTR AUC (train on synthetic labeled, test on real)")
    auc = tstr_auc(Xsyn, y_syn, X_np, y_real, cfg)
    if auc is not None:
        print(f"TSTR AUC: {auc:.3f}")

    # Conservative composition (include labeler)
    eps_total = eps_stage_a + eps_stage_b + eps_labeler
    delta_total = 3 * cfg.delta
    print(f"[Conservative composed DP] (epsilon, delta)=({eps_total:.3f}, {delta_total:.2e})")

if __name__ == "__main__":
    main(cfg)
