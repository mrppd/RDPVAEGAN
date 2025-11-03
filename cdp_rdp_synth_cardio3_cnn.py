# -*- coding: utf-8 -*-
"""
Created on Wed Jul 5 20:12:32 2023

@author: pronaya

RDP-VAEGAN for Kaggle Cardio (1D-CNN + Opacus DP) with RDP composition
"""

import os, random, math
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from opacus import PrivacyEngine
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent



# Config
# ========
@dataclass
class Config:
    data_path: str = "./cardio_train.csv"     # Kaggle CSV 

    out_csv: str = "synthetic_cardio_cnn.csv"         # z-scored/one-hot + cardio
    out_post_csv: str = "synthetic_cardio_cnn_post.csv"  # back to raw-ish units + cardio

    seed: int = 17
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256

    # CNN-VAE over 1×F signals
    latent_dim: int = 32
    enc_channels: Tuple[int, ...] = (128, 128)
    dec_channels: Tuple[int, ...] = (128, 128)
    vae_epochs: int = 20
    vae_lr: float = 1e-4
    vae_wd: float = 1e-4
    vae_target_eps: float = 3.0
    vae_beta_max: float = 0.1
    kl_warmup_epochs: int = 10
    out_scale: float = 5.0

    # DP-GAN (D is DP, G is non-DP)
    noise_dim: int = 64
    gen_fc: int = 128
    disc_channels: Tuple[int, ...] = (256, 128)
    gan_epochs: int = 25
    d_lr: float = 1e-3
    g_lr: float = 1e-3
    d_target_eps: float = 4.0

    # DP Labeler (Conv over 1×F)
    lab_channels: Tuple[int, ...] = (128, 128)
    lab_epochs: int = 8
    lab_lr: float = 1e-3
    lab_target_eps: float = 2.0

    delta: float = 1e-5
    max_grad_norm: float = 0.5
    secure_mode: bool = False  # True requires torchcsprng

    n_synth: int = 50000
    tstr_train_max: int = 50000
    tstr_test_max: int = 60000


cfg = Config()


def set_all_seeds(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_all_seeds(cfg.seed)



# Data
# ========
NUMERICS = ["age", "height", "weight", "ap_hi", "ap_lo"]
CATEGORICAL = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
TARGET = "cardio"

def load_cardio_tabular(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path)
    return df

def preprocess(df: pd.DataFrame):
    df = df.copy()
    # age days -> years if needed
    if df["age"].median() > 120: df["age"] = (df["age"]/365.25).astype(float)
    # height to cm if needed
    if df["height"].median() < 3: df["height"] = df["height"] * 100.0
    # engineered
    df["bmi"] = df["weight"] / ((df["height"]/100.0)**2 + 1e-9)
    df["pp"]  = df["ap_hi"] - df["ap_lo"]

    feat_df = df.drop(columns=[TARGET]) if TARGET in df.columns else df
    for col in ("id","ID","Id"):
        if col in feat_df.columns: feat_df = feat_df.drop(columns=[col])

    numeric_cols = [c for c in (NUMERICS+["bmi","pp"]) if c in feat_df.columns]
    cat_cols     = [c for c in CATEGORICAL if c in feat_df.columns]

    # soft-clip
    def sclip(s: pd.Series, q=0.005):
        lo, hi = s.quantile(q), s.quantile(1-q); return s.clip(lo, hi)
    for c in numeric_cols: feat_df[c] = sclip(feat_df[c].astype(float))

    feat_df[cat_cols] = feat_df[cat_cols].astype(int)
    df_oh = pd.get_dummies(feat_df, columns=cat_cols, drop_first=False)

    stats = {}
    for c in numeric_cols:
        m, s = df_oh[c].mean(), df_oh[c].std()+1e-6
        stats[c]=(m,s); df_oh[c]=(df_oh[c]-m)/s

    feature_names = list(df_oh.columns)
    meta = {"numeric_stats": stats, "oh_cols": feature_names, "numeric_cols": numeric_cols}
    return df_oh, feature_names, meta

class CardioConvDataset(Dataset):
    """Return [C=1, L=F] for Conv1d."""
    def __init__(self, X: np.ndarray):
        X = X.astype(np.float32)              # [N,F]
        self.X = torch.from_numpy(X[:, None, :])  # [N,1,F]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i]



# Models (1D-CNN) 
# =================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, padding=k//2)
        self.act = nn.LeakyReLU(0.2, inplace=False)  # Opacus-safe
    def forward(self, x):
        return self.act(self.conv(x))

class ConvVAE(nn.Module):
    def __init__(self, L: int, latent: int,
                 enc_channels: Tuple[int,...], dec_channels: Tuple[int,...], out_scale: float):
        super().__init__()
        self.L = L
        enc=[]; last=1
        for ch in enc_channels:
            enc += [ConvBlock(last, ch)]; last=ch
        self.enc = nn.Sequential(*enc)
        self.flat = last * L
        self.mu = nn.Linear(self.flat, latent)
        self.logvar = nn.Linear(self.flat, latent)
        self.fc = nn.Sequential(nn.Linear(latent, dec_channels[0]*L), nn.ReLU(inplace=False))
        dec=[]
        last = dec_channels[0]
        for ch in dec_channels[1:]:
            dec += [ConvBlock(last, ch)]; last=ch
        self.dec = nn.Sequential(*dec)
        self.out = nn.Conv1d(last, 1, kernel_size=1)
        self.tanh_vals = out_scale
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def encode(self, x):
        h = self.enc(x); h = h.reshape(x.size(0), -1)  # reshape
        mu = self.mu(h); logvar = self.logvar(h).clamp(min=-6, max=2)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = (0.5*logvar).exp(); eps = torch.randn_like(std); return mu + eps*std
    def decode(self, z):
        h = self.fc(z); h = h.reshape(z.size(0), -1, self.L)  # reshape
        h = self.dec(h) if len(self.dec)>0 else h
        return torch.tanh(self.out(h)) * self.tanh_vals
    def forward(self, x):
        mu, logvar = self.encode(x); z = self.reparameterize(mu, logvar)
        xhat = self.decode(z); return xhat, mu, logvar

def vae_loss(x, xhat, beta=0.1):
    recon = F.smooth_l1_loss(xhat, x, reduction="mean")
    return recon

class ConvGenerator(nn.Module):
    def __init__(self, noise_dim: int, L: int, gen_fc: int):
        super().__init__()
        self.L=L
        self.fc = nn.Sequential(nn.Linear(noise_dim, gen_fc*L), nn.ReLU(inplace=False))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, z):
        return self.fc(z).reshape(z.size(0), -1, self.L)

class ConvDecoderHead(nn.Module):
    """Map generator feature map [gen_fc,L] to value signal [1,L]."""
    def __init__(self, gen_fc: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(gen_fc, gen_fc, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv1d(gen_fc, 1, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, h): return self.net(h)

class ConvDiscriminator(nn.Module):
    def __init__(self, channels: Tuple[int,...]):
        super().__init__()
        layers=[]; last=1
        for ch in channels:
            layers += [nn.Conv1d(last, ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=False)]
            last = ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(last, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):  # [N,1,L]
        h = self.conv(x); h = h.mean(dim=2)
        return self.fc(h).squeeze(-1)

class ConvLabeler(nn.Module):
    def __init__(self, channels: Tuple[int,...]):
        super().__init__()
        layers=[]; last=1
        for ch in channels:
            layers += [nn.Conv1d(last, ch, 3, padding=1), nn.ReLU(inplace=False)]
            last = ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(last, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        h = self.conv(x); h = h.mean(dim=2)
        return self.fc(h).squeeze(-1)



# RDP helpers 
# =============
def _get_sample_rate(loader) -> float:
    if hasattr(loader, "sample_rate") and loader.sample_rate is not None:
        return float(loader.sample_rate)
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "sample_rate"):
        return float(loader.sampler.sample_rate)
    raise RuntimeError("Cannot find sample_rate on private loader")

def _default_orders():
    return np.concatenate([np.arange(1.25, 10.0, 0.25), np.arange(10, 256)])

def _eps_from_noise(q: float, steps: int, delta: float, noise: float) -> float:
    orders = _default_orders()
    rdp = compute_rdp(q=q, noise_multiplier=noise, steps=steps, orders=orders)
    eps, _ = get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
    return float(eps)

def _infer_noise_from_eps(q: float, steps: int, delta: float, eps_target: float) -> float:
    """
    Monotone root-find noise s.t. eps(noise) ~= eps_target.
    """
    if not math.isfinite(eps_target) or eps_target <= 0:
        return float("inf")

    # Initial bracket
    lo, hi = 1e-3, 1e3
    # Expand/contract to bracket target
    def eps_at(n): 
        try: return _eps_from_noise(q, steps, delta, n)
        except Exception: return float("inf")
    e_lo, e_hi = eps_at(lo), eps_at(hi)
    # If e_lo < target, move lower bound down
    tries = 0
    while e_lo < eps_target and lo > 1e-6 and tries < 40:
        lo *= 0.5; e_lo = eps_at(lo); tries += 1
    # If e_hi > target, move upper bound up
    tries = 0
    while e_hi > eps_target and hi < 1e6 and tries < 40:
        hi *= 2.0; e_hi = eps_at(hi); tries += 1

    # Bisection
    for _ in range(80):
        mid = math.sqrt(lo * hi)  # search in log-space
        e_mid = eps_at(mid)
        if abs(e_mid - eps_target) / max(1.0, eps_target) < 1e-3:
            return float(mid)
        if e_mid > eps_target:
            lo = mid
        else:
            hi = mid
    return float(mid)

def compose_eps_via_rdp(engines: List[PrivacyEngine], delta: float):
    """
    Compose across stages:
    - Uses sample rate & steps recorded during training
    - Infers noise by matching pe.get_epsilon(delta) if noise is unknown
    """
    orders = _default_orders()
    rdp_sum = None
    for pe in engines:
        if pe is None:
            continue
        meta = getattr(pe, "_rdp_meta", None)
        if not meta or meta.get("steps", 0) == 0:
            continue
        q = float(meta["q"])
        steps = int(meta["steps"])

        # get stage epsilon from pe
        eps_rep = pe.get_epsilon(delta)
        # infer noise that yields this epsilon, if not known
        noise = meta.get("noise", None)
        if noise is None:
            noise = _infer_noise_from_eps(q, steps, delta, eps_rep)
            meta["noise"] = float(noise)

        rdp = compute_rdp(q=q, noise_multiplier=noise, steps=steps, orders=orders)
        rdp_sum = rdp if rdp_sum is None else (rdp_sum + rdp)

    if rdp_sum is None:
        return float("inf"), float("nan")
    eps, opt_order = get_privacy_spent(orders=orders, rdp=rdp_sum, delta=delta)
    return float(eps), float(opt_order)



# DP helpers
# ============
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
    # record meta for robust composition
    pe._rdp_meta = {
        "q": _get_sample_rate(private_loader),  # sample rate
        "noise": None,                          # inferred later if not exposed
        "steps": 0,                             # DP update steps
    }
    return model, optimizer, private_loader, pe

def unwrap(m: nn.Module): return getattr(m, "_module", m)



# Training
# ==========
def train_vae_dp(model: 'ConvVAE', loader, cfg: Config):
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr, weight_decay=cfg.vae_wd)
    model, opt, priv_loader, pe = make_private_with_target_eps(
        model, opt, loader, cfg.vae_target_eps, cfg.delta, cfg.vae_epochs,
        cfg.max_grad_norm, cfg.secure_mode
    )
    model.train()
    for e in range(cfg.vae_epochs):
        beta = cfg.vae_beta_max * min(1.0, (e+1)/max(1,cfg.kl_warmup_epochs))
        losses, maes = [], []
        for xb in priv_loader:
            xb = xb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            xhat, mu, logvar = model(xb)
            recon = vae_loss(xb, xhat, beta)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta*kld
            if not torch.isfinite(loss): raise RuntimeError("NaN in VAE loss")
            loss.backward(); opt.step()
            # count DP step (noise inferred later)
            pe._rdp_meta["steps"] += 1
            losses.append(loss.item())
            with torch.no_grad():
                maes.append((xhat - xb).abs().mean().item())
        print(f"[VAE {e+1}/{cfg.vae_epochs}] loss={np.mean(losses):.6f} mae={np.mean(maes):.6f} "
              f"| eps_so_far={pe.get_epsilon(cfg.delta):.3f}, delta={cfg.delta}")
    return model, pe

@torch.no_grad()
def build_small_decoder_head(gen_fc: int): return ConvDecoderHead(gen_fc)

def train_gan_dp(dec_head: nn.Module, L: int, loader, cfg: Config):
    G = ConvGenerator(cfg.noise_dim, L, cfg.gen_fc).to(cfg.device)
    D = ConvDiscriminator(cfg.disc_channels).to(cfg.device)
    optD = torch.optim.Adam(D.parameters(), lr=cfg.d_lr, betas=(0.5,0.999))
    D, optD, priv_loader, peD = make_private_with_target_eps(
        D, optD, loader, cfg.d_target_eps, cfg.delta, cfg.gan_epochs,
        cfg.max_grad_norm, cfg.secure_mode
    )
    D_shadow = ConvDiscriminator(cfg.disc_channels).to(cfg.device)
    for p in D_shadow.parameters(): p.requires_grad=False
    D_shadow.eval(); D_shadow.load_state_dict(unwrap(D).state_dict())
    optG = torch.optim.Adam(G.parameters(), lr=cfg.g_lr, betas=(0.5,0.999))
    bce = nn.BCEWithLogitsLoss()

    for e in range(cfg.gan_epochs):
        d_losses, g_losses=[],[]
        print(f"[DEBUG] GAN epoch {e+1}/{cfg.gan_epochs}")
        for real in priv_loader:
            real = real.to(cfg.device); bs = real.size(0)
            # D step (DP)
            D.train(); optD.zero_grad(set_to_none=True)
            z = torch.randn(bs, cfg.noise_dim, device=cfg.device)
            with torch.no_grad():
                fake = dec_head(G(z))
            lossD = bce(D(real), torch.ones(bs, device=cfg.device)) + \
                    bce(D(fake), torch.zeros(bs, device=cfg.device))
            if not torch.isfinite(lossD): raise RuntimeError("NaN in D loss")
            lossD.backward(); optD.step(); d_losses.append(lossD.item())
            # count DP step
            peD._rdp_meta["steps"] += 1

            # sync shadow
            D_shadow.load_state_dict(unwrap(D).state_dict())

            # G step (non-DP)
            optG.zero_grad(set_to_none=True)
            z = torch.randn(bs, cfg.noise_dim, device=cfg.device)
            fake_g = dec_head(G(z))
            lossG = bce(D_shadow(fake_g), torch.ones(bs, device=cfg.device))
            if not torch.isfinite(lossG): raise RuntimeError("NaN in G loss")
            lossG.backward(); optG.step(); g_losses.append(lossG.item())
        print(f"[GAN {e+1}/{cfg.gan_epochs}] D={np.mean(d_losses):.4f} G={np.mean(g_losses):.4f} "
              f"| eps_so_far={peD.get_epsilon(cfg.delta):.3f}, delta={cfg.delta}")
    return G, D, peD

def train_dp_labeler_conv(X: torch.Tensor, y: np.ndarray, cfg: Config):
    dsl = torch.utils.data.TensorDataset(X, torch.from_numpy(y.astype(np.float32)))
    loader = DataLoader(dsl, batch_size=cfg.batch_size, shuffle=True)
    lab = ConvLabeler(cfg.lab_channels).to(cfg.device)
    opt = torch.optim.Adam(lab.parameters(), lr=cfg.lab_lr)
    bce = nn.BCEWithLogitsLoss()
    lab, opt, priv_loader, pe = make_private_with_target_eps(
        lab, opt, loader, cfg.lab_target_eps, cfg.delta, cfg.lab_epochs,
        cfg.max_grad_norm, cfg.secure_mode
    )
    for e in range(cfg.lab_epochs):
        losses=[]
        for xb, yb in priv_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            loss = bce(lab(xb), yb)
            if not torch.isfinite(loss): raise RuntimeError("NaN in labeler loss")
            loss.backward(); opt.step(); losses.append(loss.item())
            # count DP step
            pe._rdp_meta["steps"] += 1
        print(f"[Labeler {e+1}/{cfg.lab_epochs}] loss={np.mean(losses):.4f} "
              f"| eps_so_far={pe.get_epsilon(cfg.delta):.3f}, delta={cfg.delta}")
    return lab, pe

@torch.no_grad()
def predict_labels_conv(model: nn.Module, X: torch.Tensor, device: str):
    probs = torch.sigmoid(model(X.to(device))).cpu().numpy()
    y = (np.random.rand(len(probs)) < probs).astype(int)
    return y, probs

@torch.no_grad()
def sample_synth(G: ConvGenerator, dec_head: nn.Module, n: int, cfg: Config):
    out=[]; bs=4096
    for i in range(0, n, bs):
        z = torch.randn(min(bs, n-i), cfg.noise_dim, device=cfg.device)
        x = dec_head(G(z))
        out.append(x.detach().cpu())
    return torch.cat(out, dim=0)  # [N,1,F]

def invert_to_raw_df(Xsyn1L: torch.Tensor, feat_names: List[str], meta: Dict[str,Any]) -> pd.DataFrame:
    df = pd.DataFrame(Xsyn1L.squeeze(1).numpy(), columns=feat_names).copy()
    for col,(m,s) in meta.get("numeric_stats", {}).items():
        if col in df.columns: df[col] = df[col]*s + m
    # collapse one-hots back to categories
    bases = ["gender","cholesterol","gluc","smoke","alco","active"]
    for base in bases:
        oh = [c for c in df.columns if c.startswith(base+"_")]
        if not oh: continue
        idx = df[oh].values.argmax(axis=1)
        labels = [int(c.split("_",1)[1]) for c in oh]
        df[base] = [labels[i] for i in idx]; df.drop(columns=oh, inplace=True)
    # types & clamps
    for c in ["gender","cholesterol","gluc","smoke","alco","active","ap_hi","ap_lo"]:
        if c in df.columns: df[c] = np.rint(df[c]).astype(int)
    if "gender" in df.columns: df["gender"]=df["gender"].clip(1,2)
    if "cholesterol" in df.columns: df["cholesterol"]=df["cholesterol"].clip(1,3)
    if "gluc" in df.columns: df["gluc"]=df["gluc"].clip(1,3)
    for c in ["smoke","alco","active"]:
        if c in df.columns: df[c]=df[c].clip(0,1)
    # age back to days
    if "age" in df.columns: df["age"]=np.rint(df["age"]*365.25).astype(int)
    # recompute engineered
    for c in ["bmi","pp"]:
        if c in df.columns: df.drop(columns=[c], inplace=True)
    if {"height","weight"}<=set(df.columns):
        df["bmi"]=df["weight"]/((df["height"]/100.0)**2 + 1e-9)
    if {"ap_hi","ap_lo"}<=set(df.columns):
        df["pp"]=df["ap_hi"]-df["ap_lo"]
    # rounding
    if "height" in df.columns: df["height"]=np.rint(df["height"]).astype(int)
    for c in ["weight","bmi"]:
        if c in df.columns: df[c]=df[c].astype(float)
    order = ["age","gender","height","weight","ap_hi","ap_lo","cholesterol","gluc",
             "smoke","alco","active","bmi","pp"]
    cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    return df[cols]



# Main
# =======
def main(cfg: Config):
    df_raw = load_cardio_tabular(cfg.data_path)
    y_real = df_raw[TARGET].values.astype(np.float32)
    X_df, feat_names, meta = preprocess(df_raw)
    X_np = X_df.values.astype(np.float32)
    N, F = X_np.shape
    cfg.delta = min(cfg.delta, 1.0 / max(N, 10_000))
    print(f"Rows: {N} | Features: {F} | delta={cfg.delta:.2e}")

    ds = CardioConvDataset(X_np)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # Stage A: DP ConvVAE
    vae = ConvVAE(L=F, latent=cfg.latent_dim,
                  enc_channels=cfg.enc_channels, dec_channels=cfg.dec_channels, out_scale=cfg.out_scale).to(cfg.device)
    vae, pe_a = train_vae_dp(vae, loader, cfg)
    print(f"[Privacy after Stage A] epsilon={pe_a.get_epsilon(cfg.delta):.3f}, delta={cfg.delta}")

    # Stage B: DP-GAN (DP discriminator only)
    dec_head = build_small_decoder_head(cfg.gen_fc).to(cfg.device)
    G, D, pe_b = train_gan_dp(dec_head, L=F, loader=loader, cfg=cfg)
    print(f"[Privacy for Stage B] epsilon={pe_b.get_epsilon(cfg.delta):.3f}, delta={cfg.delta}")

    # Synthesize
    Xsyn_1L = sample_synth(G, dec_head, cfg.n_synth, cfg)  # [N,1,F]

    # Stage C: DP labeler on real (Conv)
    lab, pe_c = train_dp_labeler_conv(ds.X, y_real, cfg)
    print(f"[Privacy for Stage C] epsilon={pe_c.get_epsilon(cfg.delta):.3f}, delta={cfg.delta}")

    # Label synthetic
    y_syn, _ = predict_labels_conv(lab, Xsyn_1L, cfg.device)

    # Save z/oh with label
    df_flat = pd.DataFrame(Xsyn_1L.squeeze(1).numpy(), columns=feat_names)
    df_flat[TARGET] = y_syn
    df_flat.to_csv(cfg.out_csv, index=False)
    print(f"Saved: {cfg.out_csv} shape={df_flat.shape}")

    # Save postprocessed back to raw-ish
    df_post = invert_to_raw_df(Xsyn_1L, feat_names, meta)
    df_post[TARGET] = y_syn
    df_post.to_csv(cfg.out_post_csv, index=False)
    print(f"Saved: {cfg.out_post_csv} shape={df_post.shape}")

    # Robust RDP composition across stages A/B/C (no noise attribute needed)
    try:
        eps_tot, alpha = compose_eps_via_rdp([pe_a, pe_b, pe_c], cfg.delta)
        print(f"[RDP-composed DP] (epsilon, delta)=({eps_tot:.3f}, {cfg.delta:.2e})  [opt α={alpha:.1f}]")
    except Exception as e:
        e_sum = pe_a.get_epsilon(cfg.delta) + pe_b.get_epsilon(cfg.delta) + pe_c.get_epsilon(cfg.delta)
        print(f"[Conservative sum] epsilon≈{e_sum:.3f}, delta≈{3*cfg.delta:.2e}  (reason: {e})")


if __name__ == "__main__":
    main(cfg)
