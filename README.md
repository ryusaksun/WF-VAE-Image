# WF-VAE Image Training

This repository is now **image-first** and defaults to `WFVAE2Image` for 1024-resolution VAE training.

## Highlights

- Default model: `WFVAE2Image`
- Default train resolution: `1024`
- Default gradient accumulation: `8`
- Distributed training entrypoint: `train_image_ddp.py`
- Image reconstruction + metric scripts are under `scripts/`
- Archived video code is under `legacy/`

## Installation

```bash
git clone https://github.com/PKU-YuanGroup/WF-VAE
cd WF-VAE
conda create -n wfvae python=3.10 -y
conda activate wfvae
pip install -r requirements.txt
pip install -e .
```

## Image Training

Use the provided launcher:

```bash
bash train_wfimagevae.sh
```

See `train_wfimagevae.sh` header comments for usage examples (multi-GPU, resume, manifest split, etc.). The entrypoint `train_image_ddp.py` delegates to `train_ddp.py`.

Important defaults:

- `--model_name WFVAE2Image`
- `--resolution 1024`
- `--batch_size 4`
- `--grad_accum_steps 8`
- `--disc_cls wfimagevae.model.losses.LPIPSWithDiscriminator2D`

## Image Reconstruction

Single image:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/recon_single_image.py \
  --model_name WFVAE2Image \
  --model_config examples/wfvae2-image-1024.json \
  --ckpt_path /path/to/checkpoint-XXXX.ckpt \
  --image_path assets/gt_5544.jpg \
  --rec_path rec.jpg \
  --resolution 1024
```

Batch reconstruction:

```bash
bash examples/recon_image.sh
```

## Image Evaluation

Run LPIPS/PSNR/SSIM:

```bash
bash examples/eval_image.sh
```

The evaluator matches files by **relative path** (not basename), so nested folders are supported.

## Legacy Video Code

All video-VAE related code has been archived under `legacy/`:

- `legacy/scripts/`
- `legacy/examples/`
- `legacy/codebase/wfvideo/`

Mainline development should stay in the image pipeline (`wfimagevae/`, `scripts/`, `examples/`).

## Citation

```bibtex
@misc{li2024wfvaeenhancingvideovae,
  title={WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Model},
  author={Zongjian Li and Bin Lin and Yang Ye and Liuhan Chen and Xinhua Cheng and Shenghai Yuan and Li Yuan},
  year={2024},
  eprint={2411.17459},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.17459}
}
```

## License

Apache 2.0. See `LICENSE`.
