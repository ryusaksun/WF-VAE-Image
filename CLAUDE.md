# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WF-VAE-Image is a wavelet-driven image VAE (Variational Autoencoder) for latent diffusion models. The default model is `WFVAE2Image` targeting 1024-resolution training. Video-related code has been archived under `legacy/`; all active development is in the image pipeline.

## Environment Setup

```bash
conda create -n wfvae python=3.10 -y && conda activate wfvae
pip install -r requirements.txt
pip install -e .
```

## Common Commands

### Training (distributed)
```bash
bash train_wfimagevae.sh
```
See script header for usage (multi-GPU: `GPU=0,1,2,3 bash train_wfimagevae.sh`, resume, manifest config, etc.). Uses `torchrun` with DDP. The entrypoint `train_image_ddp.py` delegates to `train_ddp.py`.

### Single Image Reconstruction
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/recon_single_image.py \
  --model_name WFVAE2Image \
  --from_pretrained /path/to/checkpoint \
  --image_path assets/gt_5544.jpg \
  --rec_path rec.jpg \
  --resolution 1024
```

### Batch Reconstruction
```bash
bash examples/recon_image.sh
```

### Evaluation (LPIPS, PSNR, SSIM)
```bash
bash examples/eval_image.sh
```
Evaluator matches files by **relative path** (not basename), so nested directory structures are preserved.

### Save Model to HuggingFace Format
```bash
bash examples/save_hf_model.sh
```

## Testing

No pytest suite exists. Validate behavior changes by running a small-subset reconstruction/evaluation pass (`--eval_subset_size` > 0). For new reusable utilities, add tests under a `tests/` directory using `test_*.py` naming.

## Architecture

### Core Package: `wfimagevae/`

**Model registry** (`wfimagevae/model/registry.py`): Decorator-based registration — models register with `@ModelRegistry.register("WFVAE2Image")` and are retrieved via `ModelRegistry.get_model(name)`.

**Base class** (`wfimagevae/model/modeling_aebase.py`): `ImageBaseAE` inherits from diffusers' `ModelMixin` + `ConfigMixin`. `from_pretrained()` auto-detects `.ckpt` files vs HuggingFace format. When `.ckpt` is found, EMA state dict is loaded by default (disable with `NOT_USE_EMA_MODEL=1` env var).

**Main model** (`wfimagevae/model/vae/modeling_wfvae2_image.py`): `WFVAE2ImageModel` with:
- `EncoderImage`: RGB → Haar wavelet transform (12ch) → Conv → WFDownBlocks → Mid (ResNet+Attention+ResNet) → latent distribution (mean+var)
- `DecoderImage`: latent → Conv → Mid → WFUpBlocks with inverse wavelet reconstruction → RGB
- `WFDownBlockImage` / `WFUpBlockImage`: Core building blocks implementing wavelet-driven energy flow — encoder injects wavelet coefficients at each scale, decoder reconstructs them with residual connections

**Wavelet transforms** (`wfimagevae/model/modules/wavelet.py`): `HaarWaveletTransform2D` decomposes images into 12-channel coefficients (4 subbands × 3 colors); `InverseHaarWaveletTransform2D` reconstructs.

**Losses** (`wfimagevae/model/losses/`): `LPIPSWithDiscriminator2D` combines LPIPS perceptual loss with PatchGAN discriminator. Discriminator starts at step `--disc_start`.

**Datasets** (`wfimagevae/dataset/`): `TrainImageDataset` / `ValidImageDataset` recursively scan directories for images (.jpg/.png/.webp/.bmp). Images normalized to [-1, 1]. Supports manifest files (JSON lines).

### Training Flow

`train_ddp.py` orchestrates: DDP init → model + discriminator creation → optimizer setup → training loop with gradient accumulation → periodic eval (reconstruction metrics on subset) → checkpoint saving (model + optimizer + sampler state). EMA tracking is optional (`--ema`). Logging via WandB + CSV.

### Model Configuration

Model architecture is configured via JSON (`examples/wfvae2-image-1024.json`). Key parameters: `latent_dim` (8), `base_channels` ([128, 256, 512]), `encoder/decoder_energy_flow_size` (128), `norm_type` ("layernorm"), `mid_layers_type`, `scale`/`shift` for latent normalization.

## Code Style

- Python 4-space indent, PEP 8
- `snake_case` for functions/variables/CLI flags (e.g. `--grad_accum_steps`)
- `PascalCase` for classes (e.g. `WFVAE2ImageModel`, `TrainImageDataset`)
- Shell env vars uppercase (`EXP_NAME`, `TRAIN_IMAGE_DIR`)
- Reuse existing logging, arg parsing, and registry patterns before adding new abstractions
- Commits: short imperative messages, atomic and focused
