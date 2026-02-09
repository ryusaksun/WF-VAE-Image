# Repository Guidelines

## Project Structure & Module Organization
- `wfimagevae/` is the core package.
- `wfimagevae/model/` contains VAE architectures, losses, EMA, and model registry utilities.
- `wfimagevae/model/vae/modeling_wfvae2_image.py` is the main image VAE implementation.
- `wfimagevae/dataset/` provides image datasets and distributed samplers.
- `wfimagevae/eval/` implements LPIPS, PSNR, and SSIM evaluation helpers.
- `train_image_ddp.py` is the image-first multi-GPU training entrypoint (`train_ddp.py` kept for compatibility).
- `scripts/` contains image reconstruction/evaluation tools.
- `legacy/` stores archived video pipeline scripts and examples.
- `examples/` includes image-first launch scripts and configs; `assets/` is for static figures and sample media.

## Build, Test, and Development Commands
- `conda create -n wfvae python=3.10 -y && conda activate wfvae`: create the recommended environment.
- `pip install -r requirements.txt`: install runtime dependencies.
- `pip install -e .`: install this repo in editable mode for local development.
- `bash examples/train_image_ddp.sh`: launch distributed image training via `torchrun`.
- `bash examples/recon_image.sh`: reconstruct validation images in batches.
- `bash examples/eval_image.sh`: compute image reconstruction metrics (`lpips`, `psnr`, `ssim`).

## Coding Style & Naming Conventions
Use Python with 4-space indentation and follow PEP 8 style. Keep function names, variables, file names, and CLI flags in `snake_case` (for example `--grad_accum_steps`). Use `PascalCase` for classes (for example `TrainImageDataset`, `WFVAE2ImageModel`). Keep shell environment variables uppercase (`EXP_NAME`, `TRAIN_IMAGE_DIR`). Reuse existing logging, argument parsing, and module registry patterns before introducing new abstractions.

## Testing Guidelines
This repository currently uses script-driven validation rather than a dedicated `pytest` suite. For behavior changes, run at least one small-subset reconstruction/evaluation pass (`SUBSET_SIZE` > 0) and include the exact command/config in your PR notes. For new reusable utilities, add lightweight tests under a new `tests/` directory using `test_*.py` naming to support future CI.

## Commit & Pull Request Guidelines
Recent commits use short imperative messages such as `fix bug in causalcache decode` and `add auto tiling`. Keep commits focused and atomic. PRs should include: purpose, touched modules, reproducible run commands, key metric/log excerpts, and visual outputs when reconstruction quality changes. Link related issues and do not commit datasets, credentials, or large checkpoints.
