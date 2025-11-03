# RDP-VAEGAN Synthetic Data Toolkit
This repo contains end-to-end scripts to generate differentially private synthetic data using an RDP-composed pipeline. It includes ready-to-run jobs for the Kaggle Cardiovascular Disease tabular dataset and the PhysioNet Sepsis (2019) time-series dataset.


What the pipeline does
The scripts implement a three-stage synthetic data generator, based Rényi Differential Privacy (RDP) and composed at the end:

Stage A — DP-VAE (Conv1d)
Learns a low-dimensional representation and a decoder that maps latent noise → feature space. Uses Opacus for DP-SGD on the VAE.

Stage B — DP-GAN (DP discriminator only; generator is non-DP)
Trains a lightweight generator G whose output is fed through the frozen VAE decoder as a small “decoder head”. The discriminator D is trained with DP-SGD (Opacus) only, which avoids the common “grad_sample hook” instability on G.

Stage C — DP labeler (Conv1d)
Privately trains a simple classifier to assign labels to the synthetic samples (e.g., cardio or SepsisLabel). Labels are assigned by sampling from the classifier’s predicted probabilities.



Requirements
Python ≥ 3.9 (tested on 3.11 / 3.12)
PyTorch
Opacus
NumPy, pandas, scikit-learn (for evaluation), matplotlib, tqdm
