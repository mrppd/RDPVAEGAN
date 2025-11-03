# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:47:55 2023

@author: prona
"""

import os
import glob
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from opacus import PrivacyEngine                     
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.accountants.analysis.rdp import get_privacy_spent  # RDP -> (eps, delta) once at the end



# Config
# ========
@dataclass
class Config:
    # Data 
    data_dir: str = "./physionet_sepsis"  # folder containing .psv files
    recursive: bool = True                # search subfolders
    file_limit: Optional[int] = 5000      # cap number of patients loaded (None = no cap)

    # Outputs
    out_flat_csv: str = "synthetic_sepsis_flat.csv"      # model-space, includes SepsisLabel
    out_npz: str = "synthetic_sepsis_timeseries.npz"     # raw-like sequences after inversion
    out_psv_dir: str = "synthetic_psv"                   # optional .psv dump (first 100 patients)

    # General 
    seed: int = 17
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Windowing
    window_hours: int = 48        # first T hours (truncate/pad)
    drop_after_onset: bool = True # keep only hours strictly BEFORE first positive label
    include_mask: bool = True     # append missingness mask channels to features

    # DP-VAE (downsized to reduce OOM)
    latent_dim: int = 32
    vae_hidden: Tuple[int, ...] = (256, 128)
    vae_epochs: int = 15
    vae_lr: float = 1e-4
    vae_weight_decay: float = 1e-4
    vae_target_epsilon: float = 3.0
    vae_beta_max: float = 0.1
    kl_warmup_epochs: int = 8
    out_scale: float = 5.0

    # DP-GAN (downsized)
    noise_dim: int = 64
    gen_hidden: Tuple[int, ...] = (128, 128)
    disc_hidden: Tuple[int, ...] = (256, 128)
    gan_epochs: int = 20
    d_lr: float = 1e-3
    g_lr: float = 1e-3
    d_target_epsilon: float = 4.0

    # DP Labeler (predicts SepsisLabel for synthetic)
    labeler_hidden: Tuple[int, ...] = (256, 128)
    labeler_epochs: int = 8
    labeler_lr: float = 1e-3
    labeler_target_epsilon: float = 2.0

    # Privacy
    delta: float = 1e-5
    max_grad_norm: float = 0.5
    secure_mode: bool = False  # set True if torchcsprng installed

    # Batching
    batch_size: int = 64          # logical DP batch size
    physical_batch_size: int = 32 # GPU micro-batch size used by BatchMemoryManager (<= batch_size)

    # Synthesis
    n_synth: int = 20000

    # Utility checks
    tstr_train_max: int = 20000
    tstr_test_max: int = 20000


cfg = Config()


def set_all_seeds(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(cfg.seed)



# Data loading (Sepsis)
# ======================
def list_psv_files(data_dir: str, recursive: bool = True, file_limit: Optional[int] = None) -> List[str]:
    patterns = ["**/*.psv", "**/*.PSV"] if recursive else ["*.psv", "*.PSV"]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pat), recursive=recursive))
    files = sorted(set(f for f in files if os.path.isfile(f) and os.path.getsize(f) > 0))
    if file_limit is not None:
        files = files[:file_limit]
    if not files:
        raise FileNotFoundError(
            f"No .psv files found under {data_dir!r} (recursive={recursive}). "
            f"Check cfg.data_dir or set cfg.recursive accordingly."
        )
    print(f"[INFO] Found {len(files)} .psv files (recursive={recursive}) starting at {data_dir}")
    return files


def get_variable_list_from_any(files: List[str]) -> List[str]:
    for fp in files:
        try:
            cols = pd.read_csv(fp, sep="|", nrows=1).columns.tolist()
        except Exception:
            continue
        if "SepsisLabel" in cols:
            return [c for c in cols if c != "SepsisLabel"]
    raise RuntimeError("Could not find any .psv with a SepsisLabel column.")


def read_psv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep="|")


def cut_window_before_onset(df: pd.DataFrame, T: int, drop_after_onset: bool) -> pd.DataFrame:
    if not drop_after_onset:
        return df.iloc[:T]
    lbl = df["SepsisLabel"].values
    onset = np.where(lbl > 0.5)[0]
    if len(onset) > 0:
        end = max(0, onset[0])  # strictly before first positive
        df = df.iloc[:end]
    return df.iloc[:T]


def pad_to_T(df: pd.DataFrame, T: int) -> pd.DataFrame:
    if len(df) >= T:
        return df.iloc[:T].copy()
    pad_rows = T - len(df)
    pad = pd.DataFrame(np.nan, index=range(pad_rows), columns=df.columns)
    pad["SepsisLabel"] = 0.0
    return pd.concat([df, pad], ignore_index=True)


def build_patient_matrix(df: pd.DataFrame, var_list: List[str], T: int) -> Tuple[np.ndarray, np.ndarray, int]:
    vals = df[var_list].copy()
    mask = ~vals.isna()

    # Imputation: forward-fill -> back-fill; then per-variable median; whole-missing -> 0.0
    vals = vals.ffill().bfill()
    for c in var_list:
        col = vals[c]
        if col.isna().all():
            vals[c] = 0.0
        else:
            vals[c] = col.fillna(col.median())

    X = vals.values.astype(np.float32)  # [T, D]
    M = mask.values.astype(np.float32)  # [T, D]
    y = int(df["SepsisLabel"].fillna(0).max() > 0.5)  # any onset in record
    return X, M, y


def load_sepsis_dataset(
    data_dir: str,
    T: int,
    drop_after_onset: bool,
    include_mask: bool,
    recursive: bool = True,
    file_limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    files = list_psv_files(data_dir, recursive=recursive, file_limit=file_limit)
    n_files = len(files)
    print(f"[INFO] Preparing to process {n_files} .psv files... this may take a while.")

    # Progress bar
    use_tqdm = False
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(files, desc="Loading patients", unit="file")
        use_tqdm = True
    except Exception:
        iterator = files
        step = max(1, n_files // 100)

    var_list = get_variable_list_from_any(files)

    X_list, M_list, y_list = [], [], []
    for idx, fp in enumerate(iterator, 1):
        try:
            df = read_psv(fp)
        except Exception:
            if not use_tqdm and (idx % step == 0 or idx == n_files):
                print(f"[LOAD] {idx}/{n_files} ({idx / n_files:>6.2%}): unreadable file, skipped")
            continue

        if "SepsisLabel" not in df.columns:
            if not use_tqdm and (idx % step == 0 or idx == n_files):
                print(f"[LOAD] {idx}/{n_files} ({idx / n_files:>6.2%}): missing SepsisLabel, skipped")
            continue

        df = cut_window_before_onset(df, T, drop_after_onset)
        df = pad_to_T(df, T)
        X, M, y = build_patient_matrix(df, var_list, T)
        X_list.append(X)
        M_list.append(M)
        y_list.append(y)

        if not use_tqdm and (idx % step == 0 or idx == n_files):
            print(f"[LOAD] {idx}/{n_files} ({idx / n_files:>6.2%}) processed")

    if not X_list:
        raise RuntimeError("No valid patients after preprocessing (check data_dir contents).")

    X_all = np.stack(X_list, axis=0)  # [N, T, D]
    M_all = np.stack(M_list, axis=0)  # [N, T, D]
    y_all = np.array(y_list, dtype=np.int64)

    # Per-variable z-score across all patients & time
    D = X_all.shape[2]
    means = X_all.reshape(-1, D).mean(axis=0)
    stds = X_all.reshape(-1, D).std(axis=0) + 1e-6
    Xz = (X_all - means[None, None, :]) / stds[None, None, :]

    # Flatten to MLP-friendly vectors
    X_flat = Xz.reshape(Xz.shape[0], -1)
    feat_names = [f"{v}_t{t:02d}" for t in range(T) for v in var_list]

    if include_mask:
        M_flat = M_all.reshape(M_all.shape[0], -1)
        X_flat = np.concatenate([X_flat, M_flat], axis=1)
        feat_names += [f"{v}_t{t:02d}_mask" for t in range(T) for v in var_list]

    meta = {
        "T": T,
        "variables": var_list,
        "means": means,
        "stds": stds,
        "include_mask": include_mask,
        "feat_names": feat_names,
    }

    print(f"[INFO] Loaded {X_flat.shape[0]} patients. Flat feature dim = {X_flat.shape[1]}.")
    return X_flat.astype(np.float32), y_all, meta



# Dataset wrapper
# ================
class FlatSeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]



# Models (stable MLP VAE + GAN with shadow D)
# ==============================================
class TanhScaled(nn.Module):
    def __init__(self, scale: float): super().__init__(); self.scale = scale
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



# DP helpers
# ============
def make_private_with_target_eps(model, optimizer, data_loader,
                                 target_epsilon, target_delta, epochs,
                                 max_grad_norm, secure_mode):
    # NOTE: Opacus will internally choose noise multiplier for the given (eps, delta, epochs, sample_rate)
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


def extract_rdp_curve(pe: PrivacyEngine) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (orders, rdp_values) from an Opacus RDP accountant for a stage.
    This lets us compose across stages in **RDP space**.
    """
    acc = pe.accountant
    orders = getattr(acc, "orders", None)
    rdp = getattr(acc, "rdp", None)
    if orders is None:
        orders = getattr(acc, "_orders", None)
    if rdp is None:
        rdp = getattr(acc, "_rdp", None)
    if orders is None or rdp is None:
        raise RuntimeError("Cannot extract RDP curve from PrivacyEngine; Opacus internals changed?")
    orders = np.array(list(orders), dtype=float)
    rdp = np.array(list(rdp), dtype=float)
    return orders, rdp


def compose_eps_via_rdp(engines: List[PrivacyEngine], delta: float) -> Tuple[float, float]:
    """
    Compose multiple stages by summing their RDP curves, then convert once to (eps, delta).
    Returns (eps_total, opt_alpha).
    """
    if not engines:
        return 0.0, np.nan
    # Take orders from the first stage and assume the same across all (Opacus default uses a common grid)
    ord0, rdp0 = extract_rdp_curve(engines[0])
    rdp_sum = rdp0.copy()
    for pe in engines[1:]:
        ord_i, rdp_i = extract_rdp_curve(pe)
        if len(ord_i) != len(ord0) or not np.allclose(ord_i, ord0):
            raise RuntimeError("RDP orders differ between stages; cannot compose directly.")
        rdp_sum += rdp_i
    eps_total, opt_alpha = get_privacy_spent(orders=ord0, rdp=rdp_sum, delta=delta)
    return float(eps_total), float(opt_alpha)



# Training loops (with memory-safe microbatching and OOM auto-retry)
# ==================================================================
def train_vae_dp(model: VAE, loader, cfg: Config):
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr, weight_decay=cfg.vae_weight_decay)
    model, opt, priv_loader, pe = make_private_with_target_eps(
        model, opt, loader, cfg.vae_target_epsilon, cfg.delta, cfg.vae_epochs,
        cfg.max_grad_norm, cfg.secure_mode
    )

    model.train()
    for e in range(cfg.vae_epochs):
        beta = cfg.vae_beta_max * min(1.0, (e + 1) / max(1, cfg.kl_warmup_epochs))
        mae_list = []

        while True:
            try:
                with BatchMemoryManager(
                    data_loader=priv_loader,
                    max_physical_batch_size=cfg.physical_batch_size,
                    optimizer=opt,
                ) as mem_loader:
                    losses = []
                    for xb in mem_loader:
                        xb = xb.to(cfg.device)
                        opt.zero_grad(set_to_none=True)
                        xhat, mu, logvar = model(xb)
                        loss, _, _ = vae_loss(xb, xhat, mu, logvar, beta=beta)
                        if not torch.isfinite(loss): raise RuntimeError("NaN/Inf in VAE loss")
                        loss.backward(); opt.step()
                        losses.append(loss.item())
                        with torch.no_grad():
                            mae_list.append((xhat - xb).abs().mean().item())
                break  # success

            except RuntimeError as err:
                if "out of memory" in str(err).lower() and cfg.device == "cuda":
                    torch.cuda.empty_cache()
                    new_phys = max(1, cfg.physical_batch_size // 2)
                    if new_phys == cfg.physical_batch_size: raise
                    cfg.physical_batch_size = new_phys
                    print(f"[OOM][VAE] Reducing physical_batch_size to {cfg.physical_batch_size} and retrying epoch...")
                else:
                    raise

        eps = pe.get_epsilon(cfg.delta)
        print(f"[VAE {e+1}/{cfg.vae_epochs}] loss={np.mean(losses):.6f} mae={np.mean(mae_list):.6f} | "
              f"beta={beta:.3f} | eps_so_far={eps:.3f}, delta={cfg.delta}")
    return model, pe


def _freeze_all_params(m: nn.Module, freeze=True):
    for p in m.parameters(): p.requires_grad = not freeze


@torch.no_grad()
def build_clean_frozen_decoder(vae_wrapped: nn.Module, input_dim: int, cfg: Config) -> nn.Module:
    vae_base = unwrap_opacus_module(vae_wrapped)
    dec_sd = vae_base.decoder.state_dict()
    vae_fresh = VAE(input_dim, cfg.latent_dim, cfg.vae_hidden, cfg.out_scale)
    vae_fresh.decoder.load_state_dict(dec_sd)
    dec = vae_fresh.decoder
    for p in dec.parameters(): p.requires_grad = False
    dec.eval().to(cfg.device)
    return dec


def train_gan_dp(decoder: nn.Module, input_dim: int, base_loader, cfg: Config):
    decoder = decoder.to(cfg.device)
    G = Generator(cfg.noise_dim, cfg.latent_dim, cfg.gen_hidden).to(cfg.device)
    D = Discriminator(input_dim, cfg.disc_hidden).to(cfg.device)

    optD = torch.optim.Adam(D.parameters(), lr=cfg.d_lr, betas=(0.5, 0.999))
    D, optD, priv_loader, peD = make_private_with_target_eps(
        D, optD, base_loader, cfg.d_target_epsilon, cfg.delta, cfg.gan_epochs,
        cfg.max_grad_norm, cfg.secure_mode
    )
    D_shadow = Discriminator(input_dim, cfg.disc_hidden).to(cfg.device)
    _freeze_all_params(D_shadow, True); D_shadow.eval()
    D_shadow.load_state_dict(unwrap_opacus_module(D).state_dict())

    optG = torch.optim.Adam(G.parameters(), lr=cfg.g_lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    for e in range(cfg.gan_epochs):
        d_losses, g_losses = [], []
        print(f"[DEBUG] GAN epoch {e+1}/{cfg.gan_epochs}")

        while True:
            try:
                with BatchMemoryManager(
                    data_loader=priv_loader,
                    max_physical_batch_size=cfg.physical_batch_size,
                    optimizer=optD,
                ) as mem_loader:
                    for real in mem_loader:
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
                        if not torch.isfinite(lossD): raise RuntimeError("NaN/Inf in D loss")
                        lossD.backward(); optD.step()
                        d_losses.append(lossD.item())

                        # Sync shadow, then G step (non-private)
                        D_shadow.load_state_dict(unwrap_opacus_module(D).state_dict())
                        optG.zero_grad(set_to_none=True)
                        z = torch.randn(bs, cfg.noise_dim, device=cfg.device)
                        fake_g = decoder(G(z))
                        logits = D_shadow(fake_g)
                        lossG = bce(logits, torch.ones_like(logits))
                        if not torch.isfinite(lossG): raise RuntimeError("NaN/Inf in G loss")
                        lossG.backward(); optG.step()
                        g_losses.append(lossG.item())
                break

            except RuntimeError as err:
                if "out of memory" in str(err).lower() and cfg.device == "cuda":
                    torch.cuda.empty_cache()
                    new_phys = max(1, cfg.physical_batch_size // 2)
                    if new_phys == cfg.physical_batch_size: raise
                    cfg.physical_batch_size = new_phys
                    print(f"[OOM][GAN] Reducing physical_batch_size to {cfg.physical_batch_size} and retrying epoch...")
                else:
                    raise

        eps = peD.get_epsilon(cfg.delta)
        print(f"[GAN {e+1}/{cfg.gan_epochs}] D={np.mean(d_losses):.6f} G={np.mean(g_losses):.6f} | "
              f"eps_so_far={eps:.3f}, delta={cfg.delta}")
    return G, D, peD



# Post-processing: invert z-score back to raw units; rebuild sequences
# =====================================================================
def invert_flat_to_timeseries(X_flat: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    T = meta["T"]; vars_ = meta["variables"]; means = meta["means"]; stds = meta["stds"]
    D = len(vars_)
    include_mask = meta.get("include_mask", False)
    if include_mask:
        feat_dim = T * D
        X_vals = X_flat[:, :feat_dim]
    else:
        X_vals = X_flat
    Xz = X_vals.reshape(-1, T, D)  # back to [N, T, D] in z-space
    X_raw = Xz * stds[None, None, :] + means[None, None, :]
    return X_raw


def save_npz_sequences(X_raw: np.ndarray, meta: Dict[str, Any], path: str):
    np.savez_compressed(path, X=X_raw, variables=np.array(meta["variables"]), T=meta["T"])
    print(f"Saved timeseries npz: {path} with shape {X_raw.shape}")


def write_psv_folder(X_raw: np.ndarray, y_syn: np.ndarray, meta: Dict[str, Any], out_dir: str, limit: int = 100):
    os.makedirs(out_dir, exist_ok=True)
    vars_ = meta["variables"]; T = meta["T"]
    n = min(limit, X_raw.shape[0])
    for i in range(n):
        df = pd.DataFrame(X_raw[i], columns=vars_)
        df["SepsisLabel"] = int(y_syn[i])  # constant label per hour (early prediction framing)
        df.to_csv(os.path.join(out_dir, f"synthetic_{i:05d}.psv"), sep="|", index=False, float_format="%.6f")
    print(f"Wrote {n} synthetic .psv files to {out_dir}")



# DP labeler & utility checks
# ============================
def train_dp_labeler(X: np.ndarray, y: np.ndarray, cfg: Config):
    ds = FlatSeqDataset(X, y)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model = MLPLabeler(X.shape[1], cfg.labeler_hidden).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.labeler_lr)
    bce = nn.BCEWithLogitsLoss()

    model, opt, priv_loader, pe = make_private_with_target_eps(
        model, opt, loader, cfg.labeler_target_epsilon, cfg.delta, cfg.labeler_epochs,
        cfg.max_grad_norm, cfg.secure_mode
    )

    for e in range(cfg.labeler_epochs):
        while True:
            try:
                with BatchMemoryManager(
                    data_loader=priv_loader,
                    max_physical_batch_size=cfg.physical_batch_size,
                    optimizer=opt,
                ) as mem_loader:
                    losses = []
                    for xb, yb in mem_loader:
                        xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                        opt.zero_grad(set_to_none=True)
                        logits = model(xb); loss = bce(logits, yb)
                        if not torch.isfinite(loss): raise RuntimeError("NaN/Inf in labeler loss")
                        loss.backward(); opt.step()
                        losses.append(loss.item())
                break
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and cfg.device == "cuda":
                    torch.cuda.empty_cache()
                    new_phys = max(1, cfg.physical_batch_size // 2)
                    if new_phys == cfg.physical_batch_size: raise
                    cfg.physical_batch_size = new_phys
                    print(f"[OOM][Labeler] Reducing physical_batch_size to {cfg.physical_batch_size} and retrying epoch...")
                else:
                    raise

        eps = pe.get_epsilon(cfg.delta)
        print(f"[Labeler {e+1}/{cfg.labeler_epochs}] loss={np.mean(losses):.4f} | eps_so_far={eps:.3f}")
    return model, pe


@torch.no_grad()
def predict_labels(model: nn.Module, X: np.ndarray, device: str):
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    probs = torch.sigmoid(model(X_t)).cpu().numpy()
    y = (np.random.rand(len(probs)) < probs).astype(int)
    return y, probs


def ks_tests_first_hour(X_real_flat: np.ndarray, X_syn_flat: np.ndarray, meta: Dict[str, Any]):
    try:
        from scipy.stats import ks_2samp
    except Exception:
        print("[WARN] scipy not installed. Run: pip install scipy")
        return []
    T = meta["T"]; D = len(meta["variables"])
    real0 = X_real_flat[:, :D]
    syn0  = X_syn_flat[:, :D]
    res = []
    for i, v in enumerate(meta["variables"]):
        stat, p = ks_2samp(real0[:, i], syn0[:, i])
        res.append((v, float(stat), float(p)))
    return res


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
    probs = clf.predict_proba(X_real[idx_test])[:, 1]
    auc = roc_auc_score(y_real[idx_test], probs)
    return float(auc)



# Synthesis
# ==========
@torch.no_grad()
def sample_synthetic(G: Generator, decoder: nn.Module, n: int, cfg: Config) -> np.ndarray:
    G.eval(); decoder.eval()
    out, bs = [], 2048
    for i in range(0, n, bs):
        z = torch.randn(min(bs, n - i), cfg.noise_dim, device=cfg.device)
        x = decoder(G(z))
        out.append(x.detach().cpu().numpy())
    return np.concatenate(out, axis=0)



# Main (with paper-style RDP composition)
# =========================================
def main(cfg: Config):
    # Load & preprocess real data
    X_flat, y_real, meta = load_sepsis_dataset(
        cfg.data_dir, cfg.window_hours, cfg.drop_after_onset, cfg.include_mask,
        recursive=cfg.recursive, file_limit=cfg.file_limit
    )
    N = X_flat.shape[0]
    cfg.delta = min(cfg.delta, 1.0 / max(N, 10_000))
    print(f"Patients: {N} | Flat features: {X_flat.shape[1]} | delta={cfg.delta:.2e}")

    # Stage A: DP-VAE
    ds = FlatSeqDataset(X_flat)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    input_dim = X_flat.shape[1]
    vae = VAE(input_dim, cfg.latent_dim, cfg.vae_hidden, cfg.out_scale)
    vae, pe_vae = train_vae_dp(vae, loader, cfg)
    eps_a = pe_vae.get_epsilon(cfg.delta)
    print(f"[Privacy Stage A] epsilon={eps_a:.3f}, delta={cfg.delta}")
    print("[INFO] Building frozen clean decoder...")
    decoder = build_clean_frozen_decoder(vae, input_dim, cfg)
    print("[INFO] Starting DP-GAN...")

    # Stage B: DP-GAN (DP discriminator only)
    G, D, pe_disc = train_gan_dp(decoder, input_dim, loader, cfg)
    eps_b = pe_disc.get_epsilon(cfg.delta)
    print(f"[Privacy Stage B] epsilon={eps_b:.3f}, delta={cfg.delta}")

    # Synthesize in flat (z-space + optional masks)
    Xsyn_flat = sample_synthetic(G, decoder, cfg.n_synth, cfg)

    # Stage C: DP Labeler (on real flat features)
    labeler, pe_lab = train_dp_labeler(X_flat, y_real, cfg)
    eps_c = pe_lab.get_epsilon(cfg.delta)
    print(f"[Privacy Stage C (Labeler)] epsilon={eps_c:.3f}, delta={cfg.delta}")

    # Predict labels for synthetic
    y_syn, y_prob = predict_labels(labeler.to(cfg.device), Xsyn_flat, cfg.device)

    # Save model-space flat CSV with SepsisLabel
    feat_names = meta["feat_names"]
    df_flat = pd.DataFrame(Xsyn_flat, columns=feat_names)
    df_flat["SepsisLabel"] = y_syn
    df_flat.to_csv(cfg.out_flat_csv, index=False)
    print(f"Saved flat synthetic with labels: {cfg.out_flat_csv} shape={df_flat.shape}")

    # Invert back to raw units & save sequences (timeseries)
    Xsyn_raw = invert_flat_to_timeseries(Xsyn_flat, meta)
    save_npz_sequences(Xsyn_raw, meta, cfg.out_npz)

    # Write first N synthetic patients to .psv for inspection
    write_psv_folder(Xsyn_raw, y_syn, meta, cfg.out_psv_dir, limit=100)

    # Utility checks 
    print("[Utility] KS tests on hour-0 variables (z-space)")
    ks = ks_tests_first_hour(X_flat, Xsyn_flat, meta)
    if ks:
        for v, stat, p in ks[:10]:
            print(f"KS {v:>12s}: stat={stat:.3f}, p={p:.3g}")

    print("[Utility] TSTR AUC (train on synthetic labeled, test on real)")
    auc = tstr_auc(Xsyn_flat, y_syn, X_flat, y_real, cfg)
    if auc is not None:
        print(f"TSTR AUC: {auc:.3f}")

    # compose in RDP space
    try:
        eps_total, opt_alpha = compose_eps_via_rdp([pe_vae, pe_disc, pe_lab], cfg.delta)
        print(f"[RDP-composed DP] epsilon={eps_total:.3f} (opt α={opt_alpha:.2f}), delta={cfg.delta}")
    except Exception as e:
        # Fallback (conservative) 
        eps_total = eps_a + eps_b + eps_c
        print(f"[Conservative sum] epsilon≈{eps_total:.3f}, delta≈{3*cfg.delta:.2e} (looser bound)")

if __name__ == "__main__":
    main(cfg)
