# GAN vs Diffusion for Low-Data Fluorescence Microscopy: StyleGAN2-ADA vs DDPM

This repository compares two generative models trained on very small sets of fluorescence nuclei images:

| | Paper | Implementation |
|---|---|---|
| **A** | Karras et al., *Training Generative Adversarial Networks with Limited Data* (StyleGAN2-ADA, NeurIPS 2020) | NVIDIA's official code ([NVlabs/stylegan3](https://github.com/NVlabs/stylegan3), `--cfg=stylegan2`), downloaded by a setup script |
| **B** | Ho et al., *Denoising Diffusion Probabilistic Models* (NeurIPS 2020) | A PyTorch re-implementation of the paper's U-Net, training loss and sampler (`ddpm/`) |

Both models are trained on **n = 5, 25 and 100 images** from **BBBC038**. They are evaluated on two test sets:
- **BBBC039**, a different dataset;
- **unseen BBBC038 images** that no model saw in training.

All the experiments were run on Google Colab (L4 / A100 GPUs) using the notebooks in this repository.

---

## Repository layout

```
data/prepare_bbbc.py            BBBC038 -> nested training subsets (n = 5, 25, 100); BBBC039 -> val/test patches + masks
data/make_bbbc038_test.py       in-domain test set: BBBC038 fluorescence images unseen by every model
data/prepare_data.py            shared helpers (patch cropping, PNG saving); also a Cell Tracking Challenge loader
stylegan/setup_stylegan3.py     downloads NVIDIA's code and patches it for current Python / PyTorch
stylegan/train_stylegan2ada.py  Paper A training (resumable; continues from the latest snapshot)
stylegan/generate.py            samples from a StyleGAN snapshot
ddpm/ddpm.py                    Paper B: U-Net + Gaussian diffusion (L_simple, ancestral sampler)
ddpm/train.py                   Paper B training (resumable)
ddpm/sample.py                  samples from a DDPM checkpoint
eval/metrics.py                 FID / KID / precision-recall (torch-fidelity)
eval/select_best.py             picks the best snapshot by KID on validation, then scores it on test
eval/final_table.py             final comparison table across models, sizes and both test sets
run_stylegan_colab.ipynb        Colab: StyleGAN2-ADA, train 300 kimg + evaluate
run_stylegan_continue_colab.ipynb  Colab: continue one StyleGAN size to 600 kimg + re-evaluate
run_ddpm_colab.ipynb            Colab: DDPM, train + evaluate
run_final_table_colab.ipynb     Colab: builds the final results table
```

## 1. Data

Both datasets are CC0, from the Broad Bioimage Benchmark Collection.

| Dataset | Use | Download |
|---|---|---|
| [BBBC038](https://bbbc.broadinstitute.org/BBBC038) (Data Science Bowl 2018, stage 1 train) | training, and the in-domain test set | <https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip> |
| [BBBC039](https://bbbc.broadinstitute.org/BBBC039) (U2OS nuclei, Hoechst) | validation (snapshot selection), and the cross-dataset test set | [images](https://data.broadinstitute.org/bbbc/BBBC039/images.zip), [masks](https://data.broadinstitute.org/bbbc/BBBC039/masks.zip), [metadata](https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip) |

```bash
python data/prepare_bbbc.py --bbbc038 datasets/stage1_train --bbbc039 datasets/bbbc039 \
    --out datasets/bbbc_128 --crop 128 --sizes 5 25 100 --seeds 0 --heldout-frac 0
```

**BBBC038 preprocessing:**
- **Filtering.** Only the fluorescence images are kept (grayscale, dark background); brightfield and H&E images are dropped. That leaves 546 images.
- **Normalisation.** Each image is scaled to 8-bit using its own 1–99.8 percentile window.
- **Subsets.** The training subsets are **nested**: the 5 images are part of the 25, which are part of the 100. Each subset's image ids are listed in `frames.txt`.
- **Patches.** Each image is cut into 128×128 patches with stride 32, at most 64 per image.

**BBBC039** is cut into 128×128 patches, with matching nucleus masks, for its official training, validation and test splits.

**Unseen BBBC038 test set.** It's built from every fluorescence image that isn't in any training subset. The patches are made the same way as the training patches, with at most 8 per image. The set is also split by source image into halves A and B, which gives a real-vs-real reference score.

```bash
python data/make_bbbc038_test.py --bbbc038 datasets/stage1_train --data-root datasets/bbbc_128 --out datasets/bbbc038_test
```

## 2. Setup

```bash
pip install -r requirements.txt          # plus a CUDA build of PyTorch (Colab already has one)
python stylegan/setup_stylegan3.py       # clones NVIDIA's code into third_party/ and patches it
```

The setup script only makes compatibility fixes; the algorithm is unchanged:
- it replaces `pkg_resources` and `distutils`, which newer Python versions removed;
- it fixes one `Sampler.__init__` call for PyTorch 2;
- it falls back to NVIDIA's own reference (plain PyTorch) implementation when the custom CUDA kernels can't be compiled.

## 3. Paper A: StyleGAN2-ADA

```bash
python stylegan/train_stylegan2ada.py --data-root datasets/bbbc_128 --outdir runs/stylegan2ada \
    --mode scratch --sizes 5 25 100 --seeds 0 --kimg 300 --snap-kimg 50 --batch-gpu 32
# continue one size to 600 kimg in total (resumes from its latest snapshot)
python stylegan/train_stylegan2ada.py --data-root datasets/bbbc_128 --outdir runs/stylegan2ada \
    --mode scratch --sizes 25 --seeds 0 --kimg 600 --snap-kimg 50 --batch-gpu 32
```

The settings follow the paper's baseline configuration:

| Setting | Value |
|---|---|
| Config | `--cfg=stylegan2` |
| Minibatch | 64 (32 per GPU pass, with gradient accumulation) |
| Feature maps | `cbase=16384` |
| Learning rate (G and D) | 0.0025 |
| R1 γ | 1 |
| Minibatch-stddev group | 8 |
| Mapping depth | 8 |
| Mirroring | x-flips |
| Augmentation | ADA, `r_t` target 0.6 |
| Initialisation | from scratch |
| Training length | 300 kimg, then continued to 600 kimg |
| Snapshots | every ~48 kimg |

Grayscale patches are stored as RGB with the gray value copied to all three channels, and converted back to gray when sampling. If a run is interrupted, running the same command again continues from the latest valid snapshot. `--mode transfer` (fine-tuning from NVIDIA's FFHQ-256 network) is also available, but it wasn't used in these experiments.

## 4. Paper B: DDPM

```bash
for n in 5 25 100; do
  python ddpm/train.py --data-root datasets/bbbc_128 --n $n --seed 0 --steps 10000 --batch-gpu 32 \
      --ckpt-every 2500 --sample-every 2500 --outdir runs/ddpm/n$n-seed0
done
```

| Setting | Value |
|---|---|
| Diffusion | T = 1000, linear β from 1e-4 to 0.02, ε-prediction, L_simple loss |
| U-Net | ch = 128, ch_mult (1,1,2,3,4), 2 res-blocks per level, self-attention at 16×16, dropout 0.1, **85.6M parameters** |
| Optimiser | Adam, lr 5e-5, 5000-step warm-up, gradient clip 1.0 |
| Batch | 64 (32 per GPU pass, with gradient accumulation) |
| EMA | decay 0.9999 |
| Augmentation | flips and 90° rotations |
| Mixed precision | bf16 |
| Training length | 10,000 steps |
| Snapshots | EMA snapshot every 2500 steps |

At the paper's two resolutions, the U-Net reproduces the paper's parameter counts: 35.7M at 32×32 and 114M at 256×256. The 128×128 channel multipliers and learning rate sit between the paper's two settings. Final samples use the full 1000-step ancestral sampler; snapshot screening uses a 100-step respaced sampler. If training is interrupted, running the same command again resumes.

## 5. Evaluation

1. **Snapshot selection** (`eval/select_best.py`):
   - every snapshot is screened by **KID** against BBBC039 **validation** patches;
   - the best snapshot generates the final images;
   - FID, KID, precision and recall are computed against BBBC039 **test** patches.

   KID is the main metric: FID is biased when only a few hundred images are available, and KID isn't.

   ```bash
   python eval/select_best.py --kind stylegan --run runs/stylegan2ada/scratch-n25-seed0 \
       --real datasets/bbbc_128/bbbc039/validation/images --real-final datasets/bbbc_128/bbbc039/test/images \
       --n-screen 500 --n-final 1000 --every 2
   python eval/select_best.py --kind ddpm --run runs/ddpm/n25-seed0 \
       --real datasets/bbbc_128/bbbc039/validation/images --real-final datasets/bbbc_128/bbbc039/test/images \
       --n-screen 256 --n-final 300 --screen-steps 100 --every 2 --batch 128
   ```

2. **Final table** (`eval/final_table.py`). Every model generates the same number of images (300), and each set is scored on both test sets:

   | Rows | Images come from |
   |---|---|
   | StyleGAN2-ADA, 300 kimg | the last snapshot of the first training session |
   | StyleGAN2-ADA, 600 kimg | the last snapshot after continued training |
   | DDPM | the snapshot chosen in step 1 |
   | Real BBBC038 half A | real images (reference row) |

   The reference row compares real half A against real half B, which is the best score achievable in-domain. It also compares real BBBC038 against real BBBC039, which shows how much of the BBBC039 score is the dataset gap alone.

   ```bash
   python eval/final_table.py --work . --test039 datasets/bbbc_128/bbbc039/test/images --test038 datasets/bbbc038_test
   ```

   Output: `results/final_table.md`.

## 6. Running on Google Colab

1. Upload this repository as `gan_vs_diffusion.zip` to *My Drive*. Upload the prepared data as `gan_vs_diffusion_work/bbbc_128.zip`, so it isn't rebuilt every session.
2. Open each notebook in its own tab, choose a GPU runtime, and click **Runtime → Run all**.
   - Order: StyleGAN and DDPM first (they can run at the same time), then optionally the continuation notebook (one tab per size), then the final-table notebook (a T4 is enough).
3. Everything is saved to `MyDrive/gan_vs_diffusion_work/`:
   - **Checkpoints:** `runs/`.
   - **Scores:** `final_metrics.json` and `snapshot_kid.json`.
   - **Training logs:** `log.txt` / `stats.jsonl` for StyleGAN, `log.jsonl` for DDPM.
   - **Final table:** `results/`.
4. **After a disconnect:** use Run all again. Training and evaluation resume, and finished parts are skipped.

## Licences

- This repository's own code has no licence attached yet; choose one before publishing.
- NVIDIA's StyleGAN code is **not** included. `setup_stylegan3.py` downloads it, and it's covered by the [NVIDIA Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt), which allows non-commercial research use.
- BBBC038 and BBBC039 are CC0.
