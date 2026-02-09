# Data Preparation (Images)

Organize training images recursively under one root directory:

```text
train_images/
|-- class_a/
|   |-- 0001.jpg
|   |-- 0002.png
|-- class_b/
|   |-- x1.webp
|-- 10001.jpeg
```

Validation images should use a similar directory layout (`eval_images/`).

# Training (1024 default)

Update `TRAIN_IMAGE_DIR` and `EVAL_IMAGE_DIR` in `examples/train_image_ddp.sh`, then run:

```bash
bash examples/train_image_ddp.sh
```

The default setup is:
- resolution: `1024`
- batch size: `2`
- gradient accumulation: `8`
- model: `WFVAE2Image` + `examples/wfvae2-image-1024.json`
- discriminator loss: `LPIPSWithDiscriminator2D`

Key arguments:

| Arg | Usage |
|:---|:---|
| `--image_path` | `/path/to/train/images` |
| `--eval_image_path` | `/path/to/eval/images` |
| `--resolution` | image train resolution (default `1024`) |
| `--batch_size` | micro-batch size per GPU |
| `--grad_accum_steps` | gradient accumulation steps (default `8`) |
| `--model_config` | image model config JSON |
| `--pretrained_model_name_or_path` | initialize from an existing checkpoint/HF folder |
| `--resume_from_checkpoint` | resume optimizer/model/sampler states |

# Reconstruction

Run batched reconstruction:

```bash
bash examples/recon_image.sh
```

# Evaluation

Run image metrics (`LPIPS`, `PSNR`, `SSIM`):

```bash
bash examples/eval_image.sh
```

Internally, this calls `scripts/eval_image.py` with image directories:
- `--real_image_dir`
- `--generated_image_dir`

Legacy video scripts and examples are under `legacy/`.
