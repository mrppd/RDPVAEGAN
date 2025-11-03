# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 11:24:10 2023

@author: prona
"""

import os
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from opacus import PrivacyEngine
from opacus.accountants.analysis.rdp import get_privacy_spent



# Config
# ========
@dataclass
class Config:
    # Data
    data_path: str = "./cardio_train.csv"   # Kaggle CSV

    out_csv: str = "synthetic_cardio.csv"           # model-space (z-scored & one-hot) + cardio
    out_post_csv: str = "synthetic_cardio_post.csv" # original-like units + cardio

    # Training/device 
    seed: int = 17
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256

    # VAE
    latent_dim: int = 32
    vae_hidden: Tuple[int, ...] = (128, 128)
    vae_epochs: int = 20
    vae_lr: float = 1e-4
    vae_weight_decay: float = 1e-4
    vae_target_epsilon: float = 3.0
    vae_beta_max: float = 0.1
    kl_warmup_epochs: int = 10
    out_scale: float = 5.0  # decoder output bounded by tanh*out_scale

    # GAN (frozen VAE decoder; D is DP, G is non-DP)
    noise_dim: int = 64
    gen_hidden: Tuple[int, ...] = (128, 128)
    disc_hidden: Tuple[int, ...] = (256, 128)
    gan_epochs: int = 25
    d_lr: float = 1e-3
    g_lr: float = 1e-3
    d_target_epsilon: float = 4.0

    # DP Labeler (predicts cardio for synthetic)
    labeler_hidden: Tuple[int, ...] = (64,)
    labeler_epochs: int = 8
    labeler_lr: float = 1e-3
    labeler_target_epsilon: float = 2.0

    # Privacy 
    delta: float = 1e-5
    max_grad_norm: float = 0.5
    secure_mode: bool = False  # set True if torchcsprng installed

    # Synthesis
    n_synth: int = 50000

    # Utility (optional)
    tstr_train_max: int = 50000
    tstr_test_max: int = 60000


cfg = Config()


def set_all_seeds(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(cfg.seed)



# Data loading & preprocessing
# ==============================
NUMERICS = ["age", "height", "weight", "ap_hi", "ap_lo"]
CATEGORICAL = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
TARGET = "cardio"

def load_cardio_tabular(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at {path}. "
            f"Download 'cardio_train.csv' from Kaggle and update Config.data_path."
        )
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path)
    return df

def preprocess(df: pd.DataFrame):
    """
    - Drop id columns
    - Convert age in days -> years (if needed)
    - Ensure height in cm; add BMI & pulse pressure
    - One-hot categoricals; z-score numerics
    Returns (X_df, feature_names, meta)
    """
    df = df.copy()

    # Age (days -> years) if large
    if df["age"].median() > 120:
        df["age"] = (df["age"] / 365.25).astype(float)

    # Height in cm
    h = df["height"].astype(float)
    if h.median() < 3:
        df["height"] = h * 100.0

    # Engineered features
    df["bmi"] = df["weight"] / ((df["height"] / 100.0) ** 2 + 1e-9)
    df["pp"] = df["ap_hi"] - df["ap_lo"]

    # Remove label from features if present
    feat_df = df.drop(columns=[TARGET]) if TARGET in df.columns else df

    # Drop identifiers
    for col in ("id", "ID", "Id"):
        if col in feat_df.columns:
            feat_df = feat_df.drop(columns=[col])

    # Numeric & categorical columns present
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


# Models
# =======
class TanhScaled(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
    def forward(self, x): return torch.tanh(x) * self.scale

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: Tuple[int, ...], out_scale: float):
        super().__init__()
        enc, last = [], input_dim
        for w in hidden:
            enc += [nn.Linear(last, w), nn.ReLU()]; last = w
        self.encoder = nn.Sequential(*enc)
        self.mu = nn.Linear(last, latent_dim)
        self.logvar = nn.Linear(last, latent_dim)

        dec, last = [], latent_dim
        for w in reversed(hidden):
            dec += [nn.Linear(last, w), nn.ReLU()]; last = w
        dec += [nn.Linear(last, input_dim), TanhScaled(out_scale)]
        self.decoder = nn.Sequential(*dec)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.encoder(x); mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-6.0, max=2.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp(); eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x); z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z); return xhat, mu, logvar

def vae_loss(x, xhat, mu, logvar, beta=0.1):
    recon = F.smooth_l1_loss(xhat, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon.detach(), kld.detach()

class Generator(nn.Module):
    def __init__(self, noise_dim: int, latent_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers, last = [], noise_dim
        for w in hidden:
            layers += [nn.Linear(last, w), nn.ReLU()]; last = w
        layers += [nn.Linear(last, latent_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers, last = [], input_dim
        for w in hidden:
            layers += [nn.Linear(last, w), nn.LeakyReLU(0.2)]; last = w
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x).squeeze(-1)

class MLPLabeler(nn.Module):
    def __init__(self, input_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers, last = [], input_dim
        for w in hidden:
            layers += [nn.Linear(last, w), nn.ReLU()]; last = w
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x).squeeze(-1)


# DP helpers (RDP accountant + RDP-space composition)
# =====================================================
def make_private_with_target_eps(model, optimizer, data_loader,
                                 target_epsilon, target_delta, epochs,
                                 max_grad_norm, secure_mode):
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

@torch.no_grad()
def build_clean_frozen_decoder(vae_wrapped: nn.Module, input_dim: int, cfg: Config) -> nn.Module:
    vae_base = unwrap_opacus_module(vae_wrapped)
    decoder_sd = vae_base.decoder.state_dict()
    host = VAE(input_dim, cfg.latent_dim, cfg.vae_hidden, cfg.out_scale)
    host.decoder.load_state_dict(decoder_sd)
    decoder = host.decoder
    for p in decoder.parameters(): p.requires_grad = False
    decoder.eval().to(cfg.device)
    return decoder

def extract_rdp_curve(pe: PrivacyEngine):
    """
    Extract (orders, rdp_values) from the stage's RDP accountant.
    Works across Opacus versions by checking public & private attrs.
    """
    acc = pe.accountant
    orders = getattr(acc, "orders", None) or getattr(acc, "_orders", None)
    rdp = getattr(acc, "rdp", None) or getattr(acc, "_rdp", None)
    if orders is None or rdp is None:
        raise RuntimeError("Cannot extract RDP curve from PrivacyEngine.")
    orders = np.array(list(orders), dtype=float)
    rdp = np.array(list(rdp), dtype=float)
    return orders, rdp

def compose_eps_via_rdp(engines: List[PrivacyEngine], delta: float) -> Tuple[float, float]:
    """
    Paper-style composition: sum RDP(α) across stages, then convert once to (ε, δ).
    Returns (epsilon_total, optimal_alpha).
    """
    if not engines:
        return 0.0, np.nan
    ord0, rdp0 = extract_rdp_curve(engines[0])
    rdp_sum = rdp0.copy()
    for pe in engines[1:]:
        oi, ri = extract_rdp_curve(pe)
        if len(oi) != len(ord0) or not np.allclose(oi, ord0):
            raise RuntimeError("RDP orders differ between stages; cannot compose.")
        rdp_sum += ri
    eps_total, opt_alpha = get_privacy_spent(orders=ord0, rdp=rdp_sum, delta=delta)
    return float(eps_total), float(opt_alpha)



# Training
# =========
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
        print(f"[VAE {e+1}/{cfg.vae_epochs}] loss={np.mean(losses):.6f} mae={np.mean(maes):.6f} "
              f"| beta={beta:.3f} | eps_so_far={eps:.3f}, delta={cfg.delta}")
    return model, pe

def _freeze_all_params(m: nn.Module, freeze: bool = True):
    for p in m.parameters(): p.requires_grad = not freeze

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

    # Shadow (non-private) D for the generator's step
    D_shadow = Discriminator(input_dim, cfg.disc_hidden).to(cfg.device)
    _freeze_all_params(D_shadow, True); D_shadow.eval()
    D_shadow.load_state_dict(unwrap_opacus_module(D).state_dict())

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

            # Sync shadow for the G step
            D_shadow.load_state_dict(unwrap_opacus_module(D).state_dict())

            # G step (non-private, uses shadow D)
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
        print(f"[GAN {e+1}/{cfg.gan_epochs}] D={np.mean(d_losses):.6f} G={np.mean(g_losses):.6f} "
              f"| eps_so_far={eps:.3f}, delta={cfg.delta}")

    return G, D, peD


# Post-processing & utilities
# ============================
def invert_to_raw_df(Xsyn: np.ndarray, feat_names: list, meta: dict) -> pd.DataFrame:
    df = pd.DataFrame(Xsyn, columns=feat_names).copy()

    # Invert z-score
    for col, (m, s) in meta.get("numeric_stats", {}).items():
        if col in df.columns:
            df[col] = df[col] * s + m

    # Collapse one-hots
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

    # Rounding/ordering
    if "height" in df.columns:
        df["height"] = np.rint(df["height"]).astype(int)
    for c in ["weight", "bmi"]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    order = ["age","gender","height","weight","ap_hi","ap_lo",
             "cholesterol","gluc","smoke","alco","active","bmi","pp"]
    cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    return df[cols]

def ks_tests_numeric(real_df: pd.DataFrame, syn_df: pd.DataFrame, meta: dict):
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
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_syn[idx_train], y_syn[idx_train])
    probs = clf.predict_proba(X_real[idx_test])[:,1]
    auc = roc_auc_score(y_real[idx_test], probs)
    return float(auc)


# DP labeler & synthesis
# =======================
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
            if not torch.isfinite(loss): raise RuntimeError("NaN/Inf in labeler loss")
            loss.backward(); opt.step()
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
    y_syn = (np.random.rand(len(probs)) < probs).astype(int)
    return y_syn, probs

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

def save_synth(X: np.ndarray, feature_names: List[str], path: str):
    df = pd.DataFrame(X, columns=feature_names)
    df.to_csv(path, index=False)
    print(f"Saved: {path} shape={df.shape}")


# Main (paper-style RDP composition)
# ===================================
def main(cfg: Config):
    # Load & preprocess
    df_raw = load_cardio_tabular(cfg.data_path)
    y_real = df_raw[TARGET].values.astype(np.float32) if TARGET in df_raw.columns else None
    X_df, feat_names, meta = preprocess(df_raw)

    N = len(X_df)
    cfg.delta = min(cfg.delta, 1.0 / max(N, 10_000))
    print(f"Dataset rows: {N} | Features (after OH & scaling): {len(feat_names)} | delta set to {cfg.delta:.2e}")

    X_np = X_df.values.astype(np.float32)
    input_dim = X_np.shape[1]
    ds = CardioDataset(X_np)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # Stage A: DP-VAE
    vae = VAE(input_dim, cfg.latent_dim, cfg.vae_hidden, cfg.out_scale)
    vae, pe_vae = train_vae_dp(vae, loader, cfg)
    eps_a = pe_vae.get_epsilon(cfg.delta)
    print(f"[Privacy Stage A] epsilon={eps_a:.3f}, delta={cfg.delta}")

    # Stage B: DP-GAN (DP Discriminator only)
    print("[INFO] Building clean frozen decoder...")
    decoder = build_clean_frozen_decoder(vae, input_dim, cfg)
    print("[INFO] Decoder ready. Starting DP-GAN training...")
    G, D, pe_disc = train_gan_dp(decoder, input_dim, loader, cfg)
    eps_b = pe_disc.get_epsilon(cfg.delta)
    print(f"[Privacy Stage B] epsilon={eps_b:.3f}, delta={cfg.delta}")

    # Synthesize (model space)
    Xsyn = sample_synthetic(G, decoder, cfg.n_synth, cfg)

    # Stage C: DP labeler on real features
    if y_real is None:
        raise RuntimeError("Real labels not found; cannot train DP labeler.")
    labeler, pe_lab = train_dp_labeler(X_np, y_real, cfg)
    eps_c = pe_lab.get_epsilon(cfg.delta)
    print(f"[Privacy Stage C (Labeler)] epsilon={eps_c:.3f}, delta={cfg.delta}")

    # Predict labels for synthetic and overwrite default outputs to include cardio
    y_syn, _ = predict_labels(labeler.to(cfg.device), Xsyn, cfg.device)
    df_syn_lab = pd.DataFrame(Xsyn, columns=feat_names); df_syn_lab[TARGET] = y_syn
    df_syn_lab.to_csv(cfg.out_csv, index=False)
    print(f"Saved (with label): {cfg.out_csv} shape={df_syn_lab.shape}")

    df_post = invert_to_raw_df(Xsyn, feat_names, meta)
    df_post_lab = df_post.copy(); df_post_lab[TARGET] = y_syn
    df_post_lab.to_csv(cfg.out_post_csv, index=False)
    print(f"Saved (with label): {cfg.out_post_csv} shape={df_post_lab.shape}")

    # Optional utility checks
    print("[Utility] KS tests on standardized numerics")
    ks = ks_tests_numeric(X_df, pd.DataFrame(Xsyn, columns=feat_names), meta)
    if ks:
        for c, stat, p in ks:
            print(f"KS {c:>10s}: stat={stat:.3f}, p={p:.3g}")

    if y_real is not None:
        print("[Utility] TSTR AUC (train on synthetic labeled, test on real)")
        auc = tstr_auc(Xsyn, y_syn, X_np, y_real, cfg)
        if auc is not None:
            print(f"TSTR AUC: {auc:.3f}")

    # PAPER-STYLE COMPOSITION: compose in RDP and convert once
    try:
        eps_total, opt_alpha = compose_eps_via_rdp([pe_vae, pe_disc, pe_lab], cfg.delta)
        print(f"[RDP-composed DP] epsilon={eps_total:.3f} (opt α={opt_alpha:.2f}), delta={cfg.delta}")
    except Exception as e:
        print(f"[WARN] RDP curve extraction failed ({e}). Falling back to conservative epsilon sum.")
        eps_total = eps_a + eps_b + eps_c
        print(f"[Conservative sum] epsilon≈{eps_total:.3f}, delta≈{3*cfg.delta:.2e} (looser bound)")

if __name__ == "__main__":
    main(cfg)
