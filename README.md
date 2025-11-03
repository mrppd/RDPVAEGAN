# RDP-VAEGAN Synthetic Data Toolkit
This repo contains end-to-end scripts to generate differentially private synthetic data using an RDP-composed pipeline. It includes ready-to-run jobs for the Kaggle Cardiovascular Disease tabular dataset and the PhysioNet Sepsis (2019) time-series dataset.


## What the pipeline does

The scripts implement a **three-stage** synthetic data generator, based **Rényi Differential Privacy** (RDP) and composed at the end:

1. **Stage A — DP-VAE (Conv1d)**  
   Learns a low-dimensional representation and a **decoder** that maps latent noise → feature space. Uses **Opacus** for DP-SGD on the VAE.

2. **Stage B — DP-GAN** (DP **discriminator** only; **generator is non-DP**)  
   Trains a lightweight generator **G** whose output is fed through the **frozen VAE decoder** as a small “decoder head”. The **discriminator D** is trained with DP-SGD (Opacus) **only**, which avoids the common “grad_sample hook” instability on G.

3. **Stage C — DP labeler (Conv1d)**  
   Privately trains a simple classifier to assign labels to the synthetic samples (e.g., `cardio` or `SepsisLabel`). Labels are assigned by sampling from the classifier’s predicted probabilities.

## Requirements

- Python ≥ 3.9
- PyTorch  
- Opacus  
- NumPy, pandas, scikit-learn (for evaluation), matplotlib, tqdm

Install:

```bash
pip install torch opacus numpy pandas scikit-learn matplotlib tqdm
```

## Data preparation

## 1) Kaggle Cardiovascular Disease

- Place the CSV at, e.g., `./cardio_train.csv`. Some distributions use semicolon `;` separators.
- **Required target column:** `cardio` (0/1).
- The script will:
  - engineer `bmi` and `pp` (pulse pressure) features,
  - **one-hot encode** categorical fields, and
  - **z-score** numeric fields.

## 2) PhysioNet Sepsis (2019)

- Place all `.psv` files under a root folder (subfolders allowed). Example: `./physionet_sepsis/`
- The loader scans recursively and builds a fixed-length tensor per patient: **(F variables × T hours)**.
- **Imputation:** per-patient **forward fill** over time; remaining gaps filled via **global per-variable median**.
- **Scaling:** **z-score** per variable across all patients × time.



## Running the generators (Example)

## Cardio (tabular)

```bash
python dp_rdp_synth_cardio3_with_target.py
```

## Sepsis (time-series)

```bash
python dp_rdp_synth_sepsis3_with_target_mem_opt.py --root ./physionet_sepsis --recursive --max_hours 24 --batch_size 64

```


## Reference

For theoretical details, see the original paper:

Das, P.P., Tawadros, D., Wiese, L. (2023). Privacy-Preserving Medical Data Generation Using Adversarial Learning. In: Athanasopoulos, E., Mennink, B. (eds) Information Security. ISC 2023. Lecture Notes in Computer Science, vol 14411. Springer, Cham.
[https://doi.org/10.1007/978-3-031-49187-0_2](https://doi.org/10.1007/978-3-031-49187-0_2)

