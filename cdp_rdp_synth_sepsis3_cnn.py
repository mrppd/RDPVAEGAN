# -*- coding: utf-8 -*-
"""
Created on Wed Jul 5 05:34:44 2023

@author: pronaya

# RDP-VAEGAN for Sepsis with 1D-CNN + Opacus DP and RDP composition.
"""

import os, glob, math, random, argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from opacus import PrivacyEngine
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent


# Config
# =======
@dataclass
class Config:
    root: str = "./physionet_sepsis"     # root folder containing .psv files
    recursive: bool = True
    max_hours: int = 24                  # truncate/pad each patient to T hours
    batch_size: int = 64                 # keep modest to avoid CUDA OOM
    seed: int = 17
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model sizes
    latent_dim: int = 32
    enc_channels: Tuple[int, ...] = (64, 64)
    dec_channels: Tuple[int, ...] = (64, 64)
    disc_channels: Tuple[int, ...] = (128, 64)
    lab_channels:  Tuple[int, ...] = (64, 64)
    gen_fc: int = 128
    noise_dim: int = 64

    # Training
    vae_epochs: int = 15
    gan_epochs: int = 20
    lab_epochs: int = 6
    vae_lr: float = 1e-4
    vae_wd: float = 1e-4
    d_lr: float = 1e-3
    g_lr: float = 1e-3
    lab_lr: float = 1e-3
    out_scale: float = 3.0               # tanh output scaler for VAE

    # DP budgets (A + B + C ≈ total)
    vae_target_eps: float = 3.0
    d_target_eps:   float = 4.0
    lab_target_eps: float = 2.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    secure_mode: bool = False            # set True with torchcsprng installed

    # Output
    n_synth: int = 50000
    out_csv: str = "synthetic_sepsis_flat.csv"         # flattened (F*T) + SepsisLabel
    out_npy: str = "synthetic_sepsis_3d.npy"           # (N, F, T) float32
    out_meta: str = "sepsis_meta.json"                 # saved meta (feature names, scaling, etc.)


cfg = Config()



# Reproducibility
# =================
def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_all_seeds(cfg.seed)



# Data loading & preprocessing
# =============================
# Sepsis variable list (order matters; static vars as well)
PHYSIONET_VARS = [
    "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess","HCO3","FiO2","pH",
    "PaCO2","SaO2","AST","BUN","Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct",
    "Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total","TroponinI",
    "Hct","Hgb","PTT","WBC","Fibrinogen","Platelets",
    # static-ish provided in files:
    "Age","Gender","Unit1","Unit2","HospAdmTime"
]
TIME_COL = "ICULOS"
TARGET_COL = "SepsisLabel"

def list_psv_files(root: str, recursive: bool) -> List[str]:
    pattern = "**/*.psv" if recursive else "*.psv"
    return sorted(glob.glob(os.path.join(root, pattern), recursive=recursive))

def _ffill_1d_nan(arr: np.ndarray) -> np.ndarray:
    """Forward-fill a 1D array of length T with NaNs."""
    out = arr.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out

def load_all_patients(root: str, recursive: bool, max_hours: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    files = list_psv_files(root, recursive)
    print(f"[INFO] Found {len(files)} .psv files (recursive={recursive}) starting at {root}")
    if len(files) == 0:
        raise FileNotFoundError(f"No .psv files under {root}")

    F = len(PHYSIONET_VARS)
    T = int(max_hours)

    X_list = []
    y_list = []

    print(f"[INFO] Preparing to process {len(files)} .psv files... this may take a while.")
    for fp in tqdm(files, desc="Loading patients", unit="file", ascii=True):
        try:
            df = pd.read_csv(fp, sep="|")
        except Exception:
            # robust read: some files might have trailing pipes etc.
            df = pd.read_csv(fp, sep="|", engine="python")
        if TIME_COL in df.columns:
            # ICULOS starts at 1 typically; map to [0..T-1]
            df = df.sort_values(TIME_COL)
            df = df[df[TIME_COL] >= 1]
            df["hour_idx"] = (df[TIME_COL].astype(int) - 1).clip(lower=0)
        else:
            df["hour_idx"] = np.arange(len(df), dtype=int)

        # Build (F,T) with NaNs
        arr = np.full((F, T), np.nan, dtype=np.float32)

        # for each variable, place values into hours 0..T-1
        for vi, var in enumerate(PHYSIONET_VARS):
            if var not in df.columns:
                continue
            sub = df[["hour_idx", var]].dropna(subset=["hour_idx"])
            sub = sub[(sub["hour_idx"] >= 0) & (sub["hour_idx"] < T)]
            if sub.empty:
                continue
            # there can be repeated hours; take last observed
            sub = sub.groupby("hour_idx", as_index=False)[var].last()
            arr[vi, sub["hour_idx"].astype(int).values] = sub[var].astype(float).values

        # Forward-fill within patient on the time axis
        for vi in range(F):
            arr[vi, :] = _ffill_1d_nan(arr[vi, :])

        # Patient label: any sepsis flag?
        y = 0
        if TARGET_COL in df.columns:
            try:
                y = int(pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int).max())
            except Exception:
                y = int((df[TARGET_COL].fillna(0) > 0).max())

        X_list.append(arr)     # (F,T)
        y_list.append(y)

    X = np.stack(X_list, axis=0)   # (N,F,T) with NaNs
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, PHYSIONET_VARS

def impute_and_scale(X: np.ndarray, vars_list: List[str]) -> Tuple[np.ndarray, Dict[str, Tuple[float,float]]]:
    """Fill remaining NaNs by global per-variable medians across (N,T), then z-score per variable."""
    N, F, T = X.shape
    stats: Dict[str, Tuple[float,float]] = {}

    # Remaining NaNs -> global median per variable across N,T
    for f in range(F):
        med = np.nanmedian(X[:, f, :])
        if not np.isfinite(med):
            med = 0.0
        mask = np.isnan(X[:, f, :])
        X[:, f, :][mask] = med

    # z-score per variable across N,T
    Xf = X.reshape(N, F*T)
    for f, name in enumerate(vars_list):
        vals = X[:, f, :].ravel().astype(np.float64)
        m = float(vals.mean())
        s = float(vals.std() + 1e-6)
        X[:, f, :] = (X[:, f, :] - m) / s
        stats[name] = (m, s)

    return X.astype(np.float32), stats



# Dataset
# ========
class SepsisConvDataset(Dataset):
    # Returns tensor of shape [C=F, L=T]
    def __init__(self, X_3d: np.ndarray):
        self.X = torch.from_numpy(X_3d.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx]



# Models (1D-CNN; channels = variables)
# ========================================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, padding=k//2)
        self.act = nn.LeakyReLU(0.2, inplace=False)
    def forward(self, x): return self.act(self.conv(x))

class ConvVAE(nn.Module):
    def __init__(self, F: int, T: int, latent: int, enc_ch: Tuple[int,...], dec_ch: Tuple[int,...], out_scale: float):
        super().__init__()
        self.F, self.T = F, T
        enc=[]; last=F
        for ch in enc_ch:
            enc += [ConvBlock(last, ch)]; last=ch
        self.enc = nn.Sequential(*enc)
        self.flat = last * T
        self.mu = nn.Linear(self.flat, latent)
        self.logvar = nn.Linear(self.flat, latent)
        self.fc = nn.Sequential(nn.Linear(latent, dec_ch[0]*T), nn.ReLU(inplace=False))
        dec=[]
        last = dec_ch[0]
        for ch in dec_ch[1:]:
            dec += [ConvBlock(last, ch)]; last=ch
        self.dec = nn.Sequential(*dec)
        self.out = nn.Conv1d(last, F, kernel_size=1)
        self.tanh_vals = out_scale
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def encode(self, x):
        h = self.enc(x); h = h.reshape(x.size(0), -1)  # [N, flat]
        mu = self.mu(h); logvar = self.logvar(h).clamp(min=-6, max=2)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = (0.5*logvar).exp(); eps = torch.randn_like(std); return mu + eps*std
    def decode(self, z):
        h = self.fc(z); h = h.reshape(z.size(0), -1, self.T)  # [N, dec_ch0, T]
        h = self.dec(h) if len(self.dec)>0 else h
        return torch.tanh(self.out(h)) * self.tanh_vals  # [N,F,T]
    def forward(self, x):
        mu, logvar = self.encode(x); z = self.reparameterize(mu, logvar)
        xhat = self.decode(z); return xhat, mu, logvar

def vae_loss(x, xhat, beta=0.1):
    recon = F.smooth_l1_loss(xhat, x, reduction="mean")
    return recon

class ConvGenerator(nn.Module):
    def __init__(self, noise_dim: int, T: int, gen_fc: int):
        super().__init__()
        self.T=T
        self.fc = nn.Sequential(nn.Linear(noise_dim, gen_fc*T), nn.ReLU(inplace=False))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, z):  # [N,gen_fc,T]
        return self.fc(z).reshape(z.size(0), -1, self.T)

class ConvDecoderHead(nn.Module):
    """Map generator feature map [gen_fc,T] to [F,T]."""
    def __init__(self, gen_fc: int, F: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(gen_fc, gen_fc, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv1d(gen_fc, F, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, h): return self.net(h)

class ConvDiscriminator(nn.Module):
    def __init__(self, F: int, channels: Tuple[int,...]):
        super().__init__()
        layers=[]; last=F
        for ch in channels:
            layers += [nn.Conv1d(last, ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=False)]
            last = ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(last, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):  # x: [N,F,T]
        h = self.conv(x); h = h.mean(dim=2)
        return self.fc(h).squeeze(-1)

class ConvLabeler(nn.Module):
    def __init__(self, F: int, channels: Tuple[int,...]):
        super().__init__()
        layers=[]; last=F
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
# ============
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
    if not math.isfinite(eps_target) or eps_target <= 0:
        return float("inf")
    lo, hi = 1e-3, 1e3
    def eps_at(n):
        try: return _eps_from_noise(q, steps, delta, n)
        except Exception: return float("inf")
    e_lo, e_hi = eps_at(lo), eps_at(hi)
    # Adjust bracket if needed
    tries = 0
    while e_lo < eps_target and lo > 1e-6 and tries < 40:
        lo *= 0.5; e_lo = eps_at(lo); tries += 1
    tries = 0
    while e_hi > eps_target and hi < 1e6 and tries < 40:
        hi *= 2.0; e_hi = eps_at(hi); tries += 1
    # Bisection in log-space
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        e_mid = eps_at(mid)
        if abs(e_mid - eps_target) / max(1.0, eps_target) < 1e-3:
            return float(mid)
        if e_mid > eps_target:
            lo = mid
        else:
            hi = mid
    return float(mid)

def compose_eps_via_rdp(engines: List[PrivacyEngine], delta: float):
    orders = _default_orders()
    rdp_sum = None
    for pe in engines:
        if pe is None: continue
        meta = getattr(pe, "_rdp_meta", None)
        if not meta or meta.get("steps", 0) == 0:
            continue
        q = float(meta["q"]); steps = int(meta["steps"])
        eps_rep = pe.get_epsilon(delta)  # stage-reported epsilon
        noise = meta.get("noise", None)
        if noise is None:
            noise = _infer_noise_from_eps(q, steps, delta, eps_rep)
            meta["noise"] = float(noise)
        rdp = compute_rdp(q=q, noise_multiplier=noise, steps=steps, orders=orders)
        rdp_sum = rdp if rdp_sum is None else (rdp_sum + rdp)
    if rdp_sum is None:
        return float("inf"), float("nan")
    eps, alpha = get_privacy_spent(orders=orders, rdp=rdp_sum, delta=delta)
    return float(eps), float(alpha)



# DP wiring
# ===========
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
    pe._rdp_meta = {
        "q": _get_sample_rate(private_loader),
        "noise": None,        # inferred later
        "steps": 0,
    }
    return model, optimizer, private_loader, pe

def unwrap(m: nn.Module): return getattr(m, "_module", m)



# Training loops
# ================
def train_vae_dp(model: ConvVAE, loader, cfg: Config):
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr, weight_decay=cfg.vae_wd)
    model, opt, priv_loader, pe = make_private_with_target_eps(
        model, opt, loader, cfg.vae_target_eps, cfg.delta, cfg.vae_epochs,
        cfg.max_grad_norm, cfg.secure_mode
    )
    model.train()
    for e in range(cfg.vae_epochs):
        losses, maes = [], []
        for xb in priv_loader:   # xb: [N,F,T]
            xb = xb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            xhat, mu, logvar = model(xb)
            recon = vae_loss(xb, xhat, beta=0.1)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + 0.1*kld
            if not torch.isfinite(loss): raise RuntimeError("NaN in VAE loss")
            loss.backward(); opt.step()
            pe._rdp_meta["steps"] += 1
            losses.append(loss.item())
            with torch.no_grad():
                maes.append((xhat - xb).abs().mean().item())
        print(f"[VAE {e+1}/{cfg.vae_epochs}] loss={np.mean(losses):.6f} mae={np.mean(maes):.6f} "
              f"| eps_so_far={pe.get_epsilon(cfg.delta):.3f}, delta={cfg.delta}")
    return model, pe

@torch.no_grad()
def build_decoder_head(gen_fc: int, F: int) -> nn.Module:
    return ConvDecoderHead(gen_fc, F)

def train_gan_dp(dec_head: nn.Module, F: int, T: int, loader, cfg: Config):
    G = ConvGenerator(cfg.noise_dim, T, cfg.gen_fc).to(cfg.device)
    D = ConvDiscriminator(F, cfg.disc_channels).to(cfg.device)
    optD = torch.optim.Adam(D.parameters(), lr=cfg.d_lr, betas=(0.5,0.999))
    D, optD, priv_loader, peD = make_private_with_target_eps(
        D, optD, loader, cfg.d_target_eps, cfg.delta, cfg.gan_epochs,
        cfg.max_grad_norm, cfg.secure_mode
    )
    D_shadow = ConvDiscriminator(F, cfg.disc_channels).to(cfg.device)
    for p in D_shadow.parameters(): p.requires_grad=False
    D_shadow.eval(); D_shadow.load_state_dict(unwrap(D).state_dict())
    optG = torch.optim.Adam(G.parameters(), lr=cfg.g_lr, betas=(0.5,0.999))
    bce = nn.BCEWithLogitsLoss()

    for e in range(cfg.gan_epochs):
        d_losses, g_losses = [], []
        print(f"[DEBUG] GAN epoch {e+1}/{cfg.gan_epochs}")
        for real in priv_loader:
            real = real.to(cfg.device)  # [N,F,T]
            bs = real.size(0)
            # D (DP)
            D.train(); optD.zero_grad(set_to_none=True)
            z = torch.randn(bs, cfg.noise_dim, device=cfg.device)
            with torch.no_grad():
                fake = dec_head(G(z))  # [N,F,T]
            lossD = bce(D(real), torch.ones(bs, device=cfg.device)) + \
                    bce(D(fake), torch.zeros(bs, device=cfg.device))
            if not torch.isfinite(lossD): raise RuntimeError("NaN in D loss")
            lossD.backward(); optD.step(); d_losses.append(lossD.item())
            peD._rdp_meta["steps"] += 1
            # sync shadow
            D_shadow.load_state_dict(unwrap(D).state_dict())

            # G (non-DP)
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
    dsl = TensorDataset(X, torch.from_numpy(y.astype(np.float32)))
    loader = DataLoader(dsl, batch_size=cfg.batch_size, shuffle=True)
    F, T = X.shape[1], X.shape[2]
    lab = ConvLabeler(F, cfg.lab_channels).to(cfg.device)
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
    out=[]; bs=2048
    for i in range(0, n, bs):
        z = torch.randn(min(bs, n-i), cfg.noise_dim, device=cfg.device)
        x = dec_head(G(z))  # [b,F,T]
        out.append(x.detach().cpu())
    return torch.cat(out, dim=0)  # [N,F,T]



# Main
# ========
def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=cfg.root)
    parser.add_argument("--recursive", action="store_true", default=cfg.recursive)
    parser.add_argument("--max_hours", type=int, default=cfg.max_hours)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--n_synth", type=int, default=cfg.n_synth)
    parser.add_argument("--secure_mode", action="store_true", default=cfg.secure_mode)
    a = parser.parse_args(args)

    cfg.root = a.root
    cfg.recursive = a.recursive
    cfg.max_hours = a.max_hours
    cfg.batch_size = a.batch_size
    cfg.n_synth = a.n_synth
    cfg.secure_mode = a.secure_mode

    # Load
    X, y, vars_list = load_all_patients(cfg.root, cfg.recursive, cfg.max_hours)
    N, F, T = X.shape
    cfg.delta = min(cfg.delta, 1.0 / max(N, 10_000))
    print(f"Patients: {N} | Variables: {F} | Hours: {T} | delta={cfg.delta:.2e}")

    X, stats = impute_and_scale(X, vars_list)     # (N,F,T) float32
    ds = SepsisConvDataset(X)                     # returns [F,T]
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # Stage A: DP VAE
    vae = ConvVAE(F, T, cfg.latent_dim, cfg.enc_channels, cfg.dec_channels, cfg.out_scale).to(cfg.device)
    vae, pe_a = train_vae_dp(vae, loader, cfg)
    print(f"[Privacy after Stage A] epsilon={pe_a.get_epsilon(cfg.delta):.3f}, delta={cfg.delta}")

    # Stage B: DP-GAN (DP discriminator only)
    dec_head = build_decoder_head(cfg.gen_fc, F).to(cfg.device)
    G, D, pe_b = train_gan_dp(dec_head, F, T, loader, cfg)
    print(f"[Privacy for Stage B] epsilon={pe_b.get_epsilon(cfg.delta):.3f}, delta={cfg.delta}")

    # Synthesize
    Xsyn = sample_synth(G, dec_head, cfg.n_synth, cfg)      # [Ns,F,T]
    Xsyn_cpu = Xsyn.clone()

    # Stage C: DP labeler
    X_real_t = torch.from_numpy(X).to(cfg.device)           # [N,F,T]
    lab, pe_c = train_dp_labeler_conv(X_real_t, y, cfg)
    print(f"[Privacy for Stage C] epsilon={pe_c.get_epsilon(cfg.delta):.3f}, delta={cfg.delta}")

    # Label synthetic
    y_syn, _ = predict_labels_conv(lab, Xsyn, cfg.device)

    # Save
    # 3D tensor
    np.save(cfg.out_npy, Xsyn_cpu.numpy().astype(np.float32))
    print(f"Saved 3D synthetic: {cfg.out_npy} shape={Xsyn_cpu.shape}")

    # Flatten to (F*T) columns + label for eval/ML
    flat_cols = [f"{v}_t{t}" for v in vars_list for t in range(T)]
    df_flat = pd.DataFrame(Xsyn_cpu.permute(0,2,1).reshape(Xsyn_cpu.size(0), -1).numpy(), columns=flat_cols)
    df_flat[TARGET_COL] = y_syn
    df_flat.to_csv(cfg.out_csv, index=False)
    print(f"Saved flattened synthetic: {cfg.out_csv} shape={df_flat.shape}")

    # Compose ε via RDP
    try:
        eps_tot, alpha = compose_eps_via_rdp([pe_a, pe_b, pe_c], cfg.delta)
        print(f"[RDP-composed DP] (epsilon, delta)=({eps_tot:.3f}, {cfg.delta:.2e})  [opt α={alpha:.1f}]")
    except Exception as e:
        e_sum = pe_a.get_epsilon(cfg.delta) + pe_b.get_epsilon(cfg.delta) + pe_c.get_epsilon(cfg.delta)
        print(f"[Conservative sum] epsilon≈{e_sum:.3f}, delta≈{3*cfg.delta:.2e}  (reason: {e})")


if __name__ == "__main__":
    main()

