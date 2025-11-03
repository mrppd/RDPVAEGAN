# -*- coding: utf-8 -*-
"""
Created on Fri Jun 9 16:29:45 2023

@author: pronaya

"""

import os
import math
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
    vae_lr: float = 3e-4            # reduced LR for stability
    vae_target_epsilon: float = 3.0

    # GAN (frozen VAE decoder)
    noise_dim: int = 64
    gen_hidden: Tuple[int, ...] = (128, 128)
    disc_hidden: Tuple[int, ...] = (256, 128)
    gan_epochs: int = 25
    d_lr: float = 1e-3
    g_lr: float = 1e-3
    d_target_epsilon: float = 4.0

    # Privacy
    delta: float = 1e-5
    max_grad_norm: float = 0.8      # slightly tighter clipping helps
    secure_mode: bool = False       # set True if torchcsprng installed

    # Synthesis
    n_synth: int = 50000  # number of synthetic rows to output

cfg = Config()

def set_all_seeds(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(cfg.seed)


# Data loading & preprocessing 
# ===============================
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
    - Converts age in days -> years (if needed)
    - Ensures height in cm
    - Adds BMI and pulse pressure
    - One-hot encodes categoricals
    - Z-scores numeric columns
    Returns standardized feature matrix (label excluded)
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
    meta = {"numeric_stats": stats, "oh_cols": feature_names}
    return df_oh, feature_names, meta

class CardioDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i]


# Models (tabular VAE + GAN) with stability tweaks
# =================================================
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: Tuple[int, ...]):
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
        dec_layers += [nn.Linear(last, input_dim)]
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
        xhat = self.decoder(z)
        return xhat, mu, logvar

def vae_loss(x, xhat, mu, logvar, beta=0.1):
    # SmoothL1 (Huber) is more robust to occasional large residuals from DP noise
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


# DP helpers (using PrivacyEngine(accountant="rdp"))
# ===================================================
def make_private_with_target_eps(model, optimizer, data_loader,
                                 target_epsilon, target_delta, epochs,
                                 max_grad_norm, secure_mode):
    """
    Wrap model/optimizer/loader with Opacus PrivacyEngine.
    NOTE: We specify accountant="rdp" (string), not an object.
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

# Utility: unwrap an Opacus-wrapped module to its underlying torch.nn.Module
def unwrap_opacus_module(m: nn.Module) -> nn.Module:
    return getattr(m, "_module", m)

# Build a fresh, unhooked decoder with trained weights from the (wrapped) VAE
@torch.no_grad()
def build_clean_frozen_decoder(vae_wrapped: nn.Module, input_dim: int, cfg: Config) -> nn.Module:
    vae_base = unwrap_opacus_module(vae_wrapped)
    decoder_sd = vae_base.decoder.state_dict()
    vae_fresh = VAE(input_dim, cfg.latent_dim, cfg.vae_hidden)
    vae_fresh.decoder.load_state_dict(decoder_sd)
    decoder = vae_fresh.decoder
    for p in decoder.parameters():
        p.requires_grad = False
    decoder.eval()
    decoder = decoder.to(cfg.device)   # <<< add this line
    return decoder


def train_vae_dp(model: VAE, train_loader, cfg: Config):
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr)

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
        losses = []
        for x in priv_loader:
            x = x.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            xhat, mu, logvar = model(x)
            loss, _, _ = vae_loss(x, xhat, mu, logvar, beta=0.1)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf detected in VAE loss. Try reducing lr/clip or check data.")
            loss.backward()
            opt.step()
            losses.append(loss.item())

        eps = pe.get_epsilon(cfg.delta)
        print(f"[VAE {e+1}/{cfg.vae_epochs}] loss={np.mean(losses):.6f} | eps_so_far={eps:.3f}, delta={cfg.delta}")
    return model, pe

def _freeze_all_params(m: nn.Module, freeze: bool = True):
    for p in m.parameters():
        p.requires_grad = not freeze

def train_gan_dp(decoder: nn.Module, input_dim: int, train_loader, cfg: Config):
    decoder = decoder.to(cfg.device)
    G = Generator(cfg.noise_dim, cfg.latent_dim, cfg.gen_hidden).to(cfg.device)
    D = Discriminator(input_dim, cfg.disc_hidden).to(cfg.device)

    # Private discriminator
    optD = torch.optim.Adam(D.parameters(), lr=cfg.d_lr, betas=(0.5, 0.999))
    D, optD, priv_loader, peD = make_private_with_target_eps(
        D, optD, train_loader,
        target_epsilon=cfg.d_target_epsilon,
        target_delta=cfg.delta,
        epochs=cfg.gan_epochs,
        max_grad_norm=cfg.max_grad_norm,
        secure_mode=cfg.secure_mode,
    )

    # Non-private shadow copy of D (no hooks, no grads). Used ONLY for the G step forward pass.
    D_shadow = Discriminator(input_dim, cfg.disc_hidden).to(cfg.device)
    _freeze_all_params(D_shadow, freeze=True)
    D_shadow.eval()

    # Initial sync from the *underlying* torch module inside Opacus wrapper
    D_base = unwrap_opacus_module(D)
    D_shadow.load_state_dict(D_base.state_dict())

    optG = torch.optim.Adam(G.parameters(), lr=cfg.g_lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    for e in range(cfg.gan_epochs):
        d_losses, g_losses = [], []
        for real in priv_loader:
            real = real.to(cfg.device)
            bs = real.size(0)

            # D step (private, with hooks)
            D.train()
            optD.zero_grad(set_to_none=True)
            z = torch.randn(bs, cfg.noise_dim, device=cfg.device)
            with torch.no_grad():
                fake = decoder(G(z))
            lossD = bce(D(real), torch.ones(bs, device=cfg.device)) + \
                    bce(D(fake), torch.zeros(bs, device=cfg.device))
            if not torch.isfinite(lossD):
                raise RuntimeError("NaN/Inf detected in D loss.")
            lossD.backward()
            optD.step()
            d_losses.append(lossD.item())

            # Sync the shadow (no hooks, frozen) to the latest D weights
            D_base = unwrap_opacus_module(D)
            D_shadow.load_state_dict(D_base.state_dict())

            # G step (non-private)
            optG.zero_grad(set_to_none=True)
            z = torch.randn(bs, cfg.noise_dim, device=cfg.device)
            fake_g = decoder(G(z))        # decoder is CLEAN (no hooks)
            logits = D_shadow(fake_g)     # shadow D has no hooks
            lossG = bce(logits, torch.ones_like(logits))
            if not torch.isfinite(lossG):
                raise RuntimeError("NaN/Inf detected in G loss.")
            lossG.backward()
            optG.step()
            g_losses.append(lossG.item())

        eps = peD.get_epsilon(cfg.delta)
        print(f"[GAN {e+1}/{cfg.gan_epochs}] D={np.mean(d_losses):.6f} G={np.mean(g_losses):.6f} | eps_so_far={eps:.3f}, delta={cfg.delta}")

    return G, D, peD


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
# =======
def main(cfg: Config):
    # Load & preprocess
    df = load_cardio_tabular(cfg.data_path)
    X_df, feat_names, meta = preprocess(df)

    # Update delta ~ 1/N (conventional choice; keep <= provided)
    N = len(X_df)
    cfg.delta = min(cfg.delta, 1.0 / max(N, 10_000))
    print(f"Dataset rows: {N} | Features (after OH & scaling): {len(feat_names)} | delta set to {cfg.delta:.2e}")

    # Torch dataset/loader (Opacus will replace sampler internally)
    X_np = X_df.values.astype(np.float32)

    # Safety check: ensure no NaNs/Infs
    if not np.isfinite(X_np).all():
        bad = np.argwhere(~np.isfinite(X_np))
        raise ValueError(f"Found non-finite values in features at indices like {bad[:5].tolist()}.")

    ds = CardioDataset(X_np)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # Stage A: DP-VAE
    input_dim = X_np.shape[1]
    vae = VAE(input_dim, cfg.latent_dim, cfg.vae_hidden)
    vae, pe_vae = train_vae_dp(vae, loader, cfg)
    eps_stage_a = pe_vae.get_epsilon(cfg.delta)
    print(f"[Privacy after Stage A] epsilon={eps_stage_a:.3f}, delta={cfg.delta}")

    # Build CLEAN frozen decoder (no hooks) for Stage B
    decoder = build_clean_frozen_decoder(vae, input_dim, cfg)

    # Stage B: DP-GAN with shadow discriminator
    G, D, pe_disc = train_gan_dp(decoder, input_dim, loader, cfg)
    eps_stage_b = pe_disc.get_epsilon(cfg.delta)
    print(f"[Privacy for Stage B] epsilon={eps_stage_b:.3f}, delta={cfg.delta}")

    # Conservative composition 
    eps_total = eps_stage_a + eps_stage_b
    delta_total = 2 * cfg.delta
    print(f"[Conservative composed DP] (epsilon, delta)=({eps_total:.3f}, {delta_total:.2e})")

    # Synthesize 
    Xsyn = sample_synthetic(G, decoder, cfg.n_synth, cfg)
    # NOTE: numeric columns are standardized; categorical columns are one-hot and real-valued.
    # Convert each one-hot block with argmax post hoc.
    save_synth(Xsyn, feat_names, cfg)

if __name__ == "__main__":
    main(cfg)
