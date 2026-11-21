# GAN vs. Diffusion for Extreme Low-Data Fluorescence Microscopy

This project investigates how different generative modelling paradigms behave when only a very small number of microscopy images are available. It compares three approaches:

1. **StyleGAN2-ADA** — a GAN designed to improve training stability in limited-data settings.
2. **DDPM** — a pixel-space denoising diffusion probabilistic model.
3. **LDM** — a latent diffusion model that performs denoising in the compressed representation of a pretrained variational autoencoder (VAE).

The experiments use the **BBBC038 fluorescence microscopy dataset** as the primary in-domain dataset and **BBBC039** as a separate cross-dataset evaluation source for the StyleGAN2-ADA and DDPM experiments. The project also includes a detailed audit of the complete 670-image BBBC038 labelled training collection.

The central research question is:

> How do GANs, pixel-space diffusion, and latent-space diffusion compare when learning microscopy image distributions from extremely limited training data, and how do dataset heterogeneity and preprocessing affect the results?

---

## Project at a glance

| Model / component | Role | Main experimental setting |
|---|---|---|
| StyleGAN2-ADA | GAN baseline | BBBC038 subsets of 5, 25 and 100 images |
| DDPM | Pixel-space diffusion baseline | BBBC038 subsets of 5, 25 and 100 images |
| LDM | Latent-space diffusion experiment | 100 BBBC038 images, frozen pretrained VAE |
| BBBC038 dataset audit | Dataset characterisation | All 670 labelled images and their instance masks |

The StyleGAN2-ADA and DDPM experiments use a shared patch-based training and evaluation protocol. The LDM uses a separate whole-image latent-space pipeline, so its results are **not directly interchangeable** with the patch-based KID/FID/precision/recall results reported for the other two models.

---

## Repository layout

```text
data/
  prepare_bbbc.py                 BBBC038 nested subsets and BBBC039 patches
  make_bbbc038_test.py            Unseen in-domain BBBC038 test set
  prepare_data.py                 Shared data preparation helpers

stylegan/
  setup_stylegan3.py               Downloads and patches NVIDIA's StyleGAN code
  train_stylegan2ada.py            StyleGAN2-ADA training
  generate.py                      StyleGAN2-ADA sampling

ddpm/
  ddpm.py                          DDPM U-Net, diffusion process and sampler
  train.py                         DDPM training
  sample.py                        DDPM sampling

eval/
  metrics.py                       FID, KID and precision/recall
  select_best.py                   Validation-based checkpoint selection
  final_table.py                   Final cross-model comparison table

run_stylegan_colab.ipynb            Colab workflow for StyleGAN2-ADA
run_stylegan_continue_colab.ipynb   Colab workflow to continue StyleGAN training
run_ddpm_colab.ipynb                Colab workflow for DDPM
run_final_table_colab.ipynb         Colab workflow for the final comparison table

BBBC038_LDM_Final_Homogeneous_Run.ipynb
                                    Final LDM experiment and sampling workflow
```

The LDM notebook is a standalone Google Colab workflow. It expects the extracted BBBC038 dataset and the appearance-cluster audit CSV in Google Drive; see the LDM section below.

---

## 1. Datasets

Both datasets are from the Broad Bioimage Benchmark Collection.

| Dataset | Role | Download |
|---|---|---|
| [BBBC038](https://bbbc.broadinstitute.org/BBBC038) — 2018 Data Science Bowl, stage 1 training set | Main training dataset and in-domain evaluation | [stage1_train.zip](https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip) |
| [BBBC039](https://bbbc.broadinstitute.org/BBBC039) — U2OS nuclei, Hoechst stain | Validation and cross-dataset test source for the patch-based StyleGAN/DDPM benchmark | [Images](https://data.broadinstitute.org/bbbc/BBBC039/images.zip) · [Masks](https://data.broadinstitute.org/bbbc/BBBC039/masks.zip) · [Metadata](https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip) |

### BBBC038 preprocessing for StyleGAN2-ADA and DDPM

- **Filtering:** fluorescence images are retained (grayscale images with dark backgrounds), excluding brightfield and H&E images. This leaves 546 images for the patch-based benchmark.
- **Normalisation:** each image is converted to 8-bit using its own 1st–99.8th percentile intensity window.
- **Nested subsets:** the 5-image subset is contained in the 25-image subset, which is contained in the 100-image subset.
- **Patch extraction:** images are cropped into 128 × 128 patches with stride 32, with at most 64 patches per source image.

```bash
python data/prepare_bbbc.py \
    --bbbc038 datasets/stage1_train \
    --bbbc039 datasets/bbbc039 \
    --out datasets/bbbc_128 \
    --crop 128 \
    --sizes 5 25 100 \
    --seeds 0 \
    --heldout-frac 0
```

### BBBC039

BBBC039 images are converted into 128 × 128 patches with their corresponding nucleus masks, using the official training, validation and test splits.

### Unseen BBBC038 test set

The in-domain test set is constructed from fluorescence images that were not used in any of the model's training subsets. Patches are generated using the same patching procedure, with at most eight patches per source image. The set is split by source image into halves A and B to provide a real-versus-real reference.

```bash
python data/make_bbbc038_test.py \
    --bbbc038 datasets/stage1_train \
    --data-root datasets/bbbc_128 \
    --out datasets/bbbc038_test
```

### BBBC038 data used by the LDM

The LDM experiment uses the complete labelled BBBC038 training collection rather than the fluorescence-only patch subset used by the other two pipelines. The full archive contains 670 source images. Its masks are instance-level **nuclear masks**; they are not paired whole-cell and nucleus masks. Consequently, the LDM morphology analysis focuses on nuclear area, nuclei count and nuclear area fraction, rather than nucleus-to-cell area ratio.

The LDM pipeline creates an image-level split of 536 training-pool images, 67 validation images and 67 held-out test images (seed 42). It then selects 100 training images for model fitting. The final follow-up experiment selects these 100 images from the largest exploratory appearance cluster and evaluates against held-out images from that same cluster.

---

## 2. StyleGAN2-ADA

StyleGAN2-ADA is based on Karras et al., *Training Generative Adversarial Networks with Limited Data* (NeurIPS 2020).

```bash
python stylegan/train_stylegan2ada.py \
    --data-root datasets/bbbc_128 \
    --outdir runs/stylegan2ada \
    --mode scratch \
    --sizes 5 25 100 \
    --seeds 0 \
    --kimg 300 \
    --snap-kimg 50 \
    --batch-gpu 32
```

A selected run can be continued to 600 kimg:

```bash
python stylegan/train_stylegan2ada.py \
    --data-root datasets/bbbc_128 \
    --outdir runs/stylegan2ada \
    --mode scratch \
    --sizes 25 \
    --seeds 0 \
    --kimg 600 \
    --snap-kimg 50 \
    --batch-gpu 32
```

| Setting | Value |
|---|---|
| Configuration | `--cfg=stylegan2` |
| Minibatch | 64 (32 per GPU pass, with gradient accumulation) |
| Feature maps | `cbase=16384` |
| Generator/discriminator learning rate | 0.0025 |
| R1 regularisation | γ = 1 |
| Mapping network depth | 8 |
| Mirroring | Horizontal flips |
| Augmentation | ADA, target `r_t = 0.6` |
| Initialisation | From scratch |
| Training duration | 300 kimg, with selected runs continued to 600 kimg |

Grayscale patches are stored as RGB by replicating the grayscale channel and converted back to grayscale after sampling. Training can resume from the latest valid snapshot.

---

## 3. DDPM

The pixel-space DDPM follows Ho et al., *Denoising Diffusion Probabilistic Models* (NeurIPS 2020). It learns to predict Gaussian noise added to the input image over a sequence of diffusion timesteps.

```bash
for n in 5 25 100; do
  python ddpm/train.py \
      --data-root datasets/bbbc_128 \
      --n $n \
      --seed 0 \
      --steps 10000 \
      --batch-gpu 32 \
      --ckpt-every 2500 \
      --sample-every 2500 \
      --outdir runs/ddpm/n$n-seed0
done
```

| Setting | Value |
|---|---|
| Diffusion process | T = 1000; linear β from 1e-4 to 0.02 |
| Objective | ε-prediction using the simplified DDPM loss |
| U-Net | 85.6M parameters; two residual blocks per level; attention at 16 × 16 |
| Optimiser | Adam, learning rate 5e-5 |
| Warm-up | 5,000 steps |
| Gradient clipping | 1.0 |
| Batch size | 64 (32 per GPU pass, with gradient accumulation) |
| EMA | Decay 0.9999 |
| Augmentation | Horizontal flips and 90° rotations |
| Mixed precision | bfloat16 |
| Training duration | 10,000 steps |
| Checkpoints | EMA snapshots every 2,500 steps |

Final DDPM samples use the full 1,000-step ancestral sampler. A 100-step respaced sampler is used for faster checkpoint screening.

---

## 4. Latent Diffusion Model (LDM)

### Motivation

The LDM experiment investigates whether performing diffusion in a compressed latent representation can provide a practical alternative to pixel-space diffusion when microscopy training data are scarce. Instead of applying the denoising process directly to RGB pixels, the input image is encoded into a lower-resolution latent tensor by a pretrained VAE. A compact U-Net learns to predict noise in this latent space, and the VAE decoder converts sampled latents back into images.

### Architecture and implementation

The implementation uses:

- **Pretrained autoencoder:** `stabilityai/sd-vae-ft-mse`, loaded using Hugging Face Diffusers.
- **Frozen VAE:** the encoder and decoder weights are not updated during LDM training.
- **Input resolution:** 128 × 128 RGB.
- **Latent representation:** 4 × 16 × 16, using the VAE posterior mean and the conventional latent scaling factor 0.18215.
- **Latent standardisation:** per-channel mean and standard deviation computed from training latents only; the same statistics are applied to validation/test latents and reversed before decoding.
- **Denoiser:** compact residual U-Net with sinusoidal timestep embeddings, approximately 2.0 million trainable parameters.
- **Diffusion objective:** noise prediction with mean-squared error.
- **Noise schedule:** 200-step cosine schedule.
- **Optimiser:** AdamW, learning rate 1e-4, weight decay 1e-4.
- **Training:** mixed precision, gradient clipping and exponential moving average (EMA) of model weights.
- **Sampling:** deterministic 50-step DDIM-style sampling, followed by VAE decoding.

### Final homogeneous-subset follow-up

A final follow-up experiment was run on a Tesla T4 GPU in Google Colab. The 670-image dataset was split by source image into 536 training-pool images, 67 validation images and 67 held-out test images. The largest exploratory appearance cluster was cluster 0. It contained 303 images in the training pool; 100 were selected for training. The corresponding same-cluster validation and test sets contained 32 and 37 images, respectively.

This run introduced two changes relative to the earlier random-subset baseline:

1. **Appearance-homogeneous subset:** the 100 training images were selected from the largest K-means appearance cluster identified by the dataset audit.
2. **Aspect-ratio-preserving preprocessing:** images were resized to fit within 128 × 128 and padded, rather than stretched directly into a square. Image padding used the median corner colour; mask padding used zero.

The model was trained for 5,000 steps. Validation used fixed noise and timestep values at each checkpoint, and the best EMA checkpoint was selected for generation.

### LDM results

| Measure | Final follow-up result |
|---|---:|
| BBBC038 images audited | 670 |
| Training images | 100 |
| Same-cluster validation images | 32 |
| Same-cluster held-out test images | 37 |
| VAE reconstruction MAE | 0.00863 |
| Best fixed-noise EMA validation MSE | 0.38068 at step 1,000 |
| Final fixed-noise EMA validation MSE | 0.69644 at step 5,000 |
| Generated samples | 100 |

Training noise-prediction loss continued to decrease, while fixed validation MSE reached its minimum at step 1,000 and subsequently increased. This indicates a substantial generalisation gap and is consistent with overfitting to the small training subset. The VAE reconstruction MAE was low, but this pixel-level metric should not be interpreted as proof that all fine biological structures were preserved.

The generated samples still did not provide convincing separated nuclear structures. The final run's exploratory morphology distances were:

| Feature | Hellinger distance |
|---|---:|
| Nuclear area | 0.27286 |
| Connected-component count | 0.53378 |
| Nuclear area fraction | 0.97995 |

These are **exploratory diagnostics**, not definitive biological-fidelity scores. Real images were measured using manually annotated nuclear instance masks, while generated images were segmented using Otsu thresholding and connected-component analysis. Since the segmentation procedures differ, the resulting distances should be interpreted cautiously. This follow-up reports three morphology features; it should not be compared numerically with the five-feature baseline aggregate as though the metrics were identical.

### Interpretation

The LDM experiment demonstrates that the pretrained VAE can reconstruct the inspected microscopy images with a low pixel-wise error, but the compact latent denoiser did not learn a sufficiently accurate distribution to generate convincing nuclear morphology. The validation curve suggests overfitting/generalisation failure despite improved training loss.

The follow-up also does not isolate the effect of homogeneous subset selection from the effect of aspect-ratio-preserving preprocessing, because both were changed together. The results therefore do not establish that dataset heterogeneity alone caused the weak generation quality. Other plausible contributors include the small training set, the natural-image domain of the pretrained VAE, limited denoiser capacity and the difference between real and generated segmentation procedures.

### Running the LDM notebook

Open `BBBC038_LDM_Final_Homogeneous_Run.ipynb` in Google Colab and select a GPU runtime. The notebook expects the following Google Drive structure:

```text
My Drive/
└── BBBC038_LDM/
    ├── data_extracted_full/
    │   └── stage1_train/
    └── dataset_audit/
        └── appearance_clusters.csv
```

The notebook writes its outputs to:

```text
My Drive/BBBC038_LDM/ldm_outputs/final_homogeneous_cluster100/
```

This includes the latest and best EMA checkpoints, latent caches, generated samples, montage, VAE reconstruction comparison, training/validation curves, split manifest and run metadata.

---

## 5. Evaluation

### StyleGAN2-ADA and DDPM

The patch-based benchmark uses KID, FID, precision and recall, computed with `torch-fidelity`.

1. **Snapshot selection:** snapshots are screened by KID against BBBC039 validation patches.
2. **Final evaluation:** the selected checkpoints generate a common number of samples, which are evaluated against:
   - BBBC039 test patches (cross-dataset evaluation);
   - unseen BBBC038 patches (in-domain evaluation).
3. **Real-versus-real reference:** unseen BBBC038 half A is compared with half B and with BBBC039 to help separate model error from the dataset gap.

KID is emphasised because it is an unbiased estimator of squared maximum mean discrepancy and is generally more suitable than FID when sample counts are small. FID is still reported for completeness.

Example commands:

```bash
python eval/select_best.py \
    --kind stylegan \
    --run runs/stylegan2ada/scratch-n25-seed0 \
    --real datasets/bbbc_128/bbbc039/validation/images \
    --real-final datasets/bbbc_128/bbbc039/test/images \
    --n-screen 500 \
    --n-final 1000 \
    --every 2

python eval/select_best.py \
    --kind ddpm \
    --run runs/ddpm/n25-seed0 \
    --real datasets/bbbc_128/bbbc039/validation/images \
    --real-final datasets/bbbc_128/bbbc039/test/images \
    --n-screen 256 \
    --n-final 300 \
    --screen-steps 100 \
    --every 2 \
    --batch 128
```

Build the final comparison table:

```bash
python eval/final_table.py \
    --work . \
    --test039 datasets/bbbc_128/bbbc039/test/images \
    --test038 datasets/bbbc038_test
```

Output: `results/final_table.md`.

### LDM morphology evaluation

The LDM notebook computes exploratory histogram-based distances between real and generated nuclear morphology distributions. The current implementation uses the Hellinger form:

\[
H(P,Q) = \sqrt{1-\sum_b\sqrt{P(b)Q(b)}}
\]

The values are useful as diagnostics, but the current real/generated segmentation mismatch means they should not be treated as a fully validated MDD metric. The LDM morphology results are reported separately from the KID/FID/precision/recall benchmark.

---

## 6. Main findings

The patch-based StyleGAN2-ADA and DDPM experiments show that the pixel-space DDPM generally outperformed StyleGAN2-ADA on the in-domain BBBC038 evaluation for training subsets of 25 and 100 images. StyleGAN2-ADA improved with longer training, but its precision remained low in the tested settings. The cross-dataset BBBC039 evaluation also demonstrated that a model can score well on a different dataset while failing to represent the diversity of its own training domain.

The LDM experiment had a different outcome: its frozen pretrained VAE reconstructed images reasonably well, but the compact latent denoiser did not produce convincing nuclear samples. Its training/validation divergence suggests that learning a useful latent distribution from only 100 images remains difficult in this setup.

The dataset audit found substantial variation across BBBC038 images and nuclear morphology. It also identified potential image-shape distortion from direct square resizing. These findings motivate more careful dataset selection, aspect-ratio-preserving preprocessing and evaluation protocols that distinguish visual similarity from biological fidelity.

The experiments were conducted with limited seeds and compute. Results should therefore be interpreted as evidence from the tested configurations, not as universal rankings of GANs, DDPMs and LDMs.

---

## 7. Full BBBC038 dataset audit

The audit examined all 670 labelled BBBC038 images and identified 29,461 individual nuclear instance-mask files. A masks directory was present for every image.

Per-image measurements showed substantial variation:

| Feature | Mean ± standard deviation | Range |
|---|---:|---:|
| Nuclei count | 43.97 ± 47.96 | 1–375 |
| Nuclear area fraction | 0.1392 ± 0.1107 | 0.000288–0.5920 |
| Mean nuclear area | 560.48 ± 634.75 px² | 26–7,244 |
| Mean eccentricity | 0.6970 ± 0.0860 | 0.434–0.940 |

K-means clustering (K = 5) on PCA-reduced appearance and image-quality features produced imbalanced exploratory clusters. The clusters describe groups in the selected feature space; they are not verified biological classes or acquisition modalities.

Direct resizing to 128 × 128 was estimated to produce more than 10% unequal horizontal-versus-vertical scaling in 229 images (34.2%). This motivated the aspect-ratio-preserving resize-and-pad preprocessing in the final LDM follow-up.

The audit also found that the BBBC038 instance annotations describe nuclei rather than paired whole-cell and nucleus boundaries. Therefore, nucleus-to-cell area ratio is not available from these masks.

---

## 8. Running on Google Colab

1. Upload or clone the repository into Google Drive.
2. Open the relevant notebook in Colab and select an available GPU runtime.
3. Run the model-specific notebook from top to bottom.
4. Keep checkpoints and outputs on Drive so that they survive runtime disconnections.

For the patch-based StyleGAN2-ADA and DDPM notebooks, follow their respective data and output paths. For the LDM notebook, use the Drive directory structure shown in Section 4.

GPU availability and runtime duration depend on the Colab account and current resource availability. Training runs should save checkpoints regularly.

---

## References

1. T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, and T. Aila, “Training Generative Adversarial Networks with Limited Data,” NeurIPS, 2020.
2. J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” NeurIPS, 2020.
3. R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-Resolution Image Synthesis with Latent Diffusion Models,” CVPR, 2022.
4. A. Q. Nichol and P. Dhariwal, “Improved Denoising Diffusion Probabilistic Models,” ICML, 2021.
5. J. Song, C. Meng, and S. Ermon, “Denoising Diffusion Implicit Models,” ICLR, 2021.
6. D. Eschweiler et al., “Denoising diffusion probabilistic models for generation of realistic fully-annotated microscopy image datasets,” PLOS Computational Biology, 2024.
7. J. C. Caicedo et al., “Nucleus segmentation across imaging experiments: The 2018 Data Science Bowl,” Nature Methods, 2019.

---

## Licences

- This repository's own code does not currently specify a licence; choose one before public release.
- NVIDIA's StyleGAN implementation is downloaded by the setup script and remains subject to NVIDIA's source-code licence.
- BBBC038 and BBBC039 dataset terms should be checked at their respective Broad Bioimage Benchmark Collection pages before redistribution.
