import argparse
from contextlib import nullcontext
import csv
from datetime import datetime
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
import tqdm
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from wfimagevae.dataset.ddp_sampler import CustomDistributedSampler
from wfimagevae.dataset.image_dataset import (
    BaseImageDataset,
    TrainImageDataset,
    ValidImageDataset,
)
from wfimagevae.eval.cal_lpips import calculate_lpips
from wfimagevae.eval.cal_psnr import calculate_psnr
from wfimagevae.eval.cal_ssim import calculate_ssim
from wfimagevae.model import *
from wfimagevae.model.ema_model import EMA
from wfimagevae.model.utils.module_utils import resolve_str_to_obj

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def setup_logger(rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        f"[rank{rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(stream_handler)
    return logger


def total_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total / 1e6)


def set_train(modules):
    for module in modules:
        module.train()


def set_eval(modules):
    for module in modules:
        module.eval()


def set_requires_grad(module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)


def save_checkpoint(
    epoch,
    current_step,
    optimizer_state,
    state_dict,
    scaler_state,
    sampler_state,
    checkpoint_dir,
    filename="checkpoint.ckpt",
    ema_state_dict=None,
):
    if ema_state_dict is None:
        ema_state_dict = {}

    filepath = checkpoint_dir / Path(filename)
    torch.save(
        {
            "epoch": epoch,
            "current_step": current_step,
            "optimizer_state": optimizer_state,
            "state_dict": state_dict,
            "ema_state_dict": ema_state_dict,
            "scaler_state": scaler_state,
            "sampler_state": sampler_state,
        },
        filepath,
    )
    return filepath


def warmup_dataset_index_cache(
    image_path: str,
    eval_image_path: str,
    use_manifest: bool,
    global_rank: int,
    logger,
):
    if use_manifest:
        return

    if global_rank == 0:
        try:
            BaseImageDataset(
                image_folder=image_path,
                cache_file="idx_image.pkl",
                is_main_process=True,
                use_manifest=False,
            )
            BaseImageDataset(
                image_folder=eval_image_path,
                cache_file="idx_image_eval.pkl",
                is_main_process=True,
                use_manifest=False,
            )
        except Exception as exc:
            logger.warning(f"Dataset index cache warmup failed: {exc}")
    dist.barrier()


def get_exp_name(args):
    return (
        f"{args.exp_name}-lr{args.lr:.2e}-bs{args.batch_size}-"
        f"ga{args.grad_accum_steps}-rs{args.resolution}"
    )


CSV_FIELDS = [
    "timestamp",
    "epoch",
    "batch_idx",
    "global_step",
    "phase",
    "is_ema",
    "train_total_loss",
    "train_logvar",
    "train_kl_loss",
    "train_nll_loss",
    "train_rec_loss",
    "train_wl_loss",
    "train_distill_loss",
    "train_d_weight",
    "train_disc_factor",
    "train_g_loss",
    "train_d_loss",
    "train_disc_loss",
    "train_logits_real",
    "train_logits_fake",
    "train_latents_std",
    "val_psnr",
    "val_ssim",
    "val_lpips",
]

LIVE_PLOT_FIELDS = [field for field in CSV_FIELDS if field.startswith("train_")]

# Page 1: key losses — each metric gets its own subplot
LIVE_PLOT_PAGE1_TITLE = "Key Losses"
LIVE_PLOT_PAGE1_KEYS = [
    "train_total_loss",
    "train_rec_loss",
    "train_kl_loss",
    "train_g_loss",
    "train_wl_loss",
    "train_disc_loss",
]

# Page 2: other metrics — each metric gets its own subplot
LIVE_PLOT_PAGE2_TITLE = "Other Metrics"
LIVE_PLOT_PAGE2_KEYS = [
    "train_nll_loss",
    "train_d_loss",
    "train_d_weight",
    "train_disc_factor",
    "train_logvar",
    "train_latents_std",
    "train_logits_real",
    "train_logits_fake",
    "train_distill_loss",
]


def to_scalar(value):
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return value


def to_plot_value(value):
    scalar = to_scalar(value)
    if scalar == "" or scalar is None:
        return float("nan")
    try:
        return float(scalar)
    except Exception:
        return float("nan")


class TrainingCSVLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        self.file = open(self.path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        if (not file_exists) or self.path.stat().st_size == 0:
            self.writer.writeheader()
            self.file.flush()

    def log(self, row: dict):
        output = {key: "" for key in CSV_FIELDS}
        output.update(row)
        self.writer.writerow(output)
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


class LiveLossPlotter:
    def __init__(self, path: Path, history: int = 0, dpi: int = 150):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.history = max(0, int(history))
        self.dpi = int(dpi)
        self.steps = []
        self.values = {key: [] for key in LIVE_PLOT_FIELDS}

    def load_history_from_csv(self, csv_path: Path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    phase = row.get("phase", "")
                    if phase and not phase.startswith("train"):
                        continue
                    step_str = row.get("global_step", "")
                    if not step_str:
                        continue
                    try:
                        step = int(step_str)
                    except (ValueError, TypeError):
                        continue
                    self.steps.append(step)
                    for key in LIVE_PLOT_FIELDS:
                        raw = row.get(key, "")
                        if raw == "" or raw is None:
                            self.values[key].append(float("nan"))
                        else:
                            try:
                                self.values[key].append(float(raw))
                            except (ValueError, TypeError):
                                self.values[key].append(float("nan"))
        except Exception:
            return

    def update(self, step: int, metrics: dict):
        self.steps.append(int(step))
        for key in LIVE_PLOT_FIELDS:
            self.values[key].append(to_plot_value(metrics.get(key)))

        if self.history > 0 and len(self.steps) > self.history:
            keep_from = len(self.steps) - self.history
            self.steps = self.steps[keep_from:]
            for key in LIVE_PLOT_FIELDS:
                self.values[key] = self.values[key][keep_from:]

    def _render_page(self, keys, page_title, save_path):
        active_keys = []
        for key in keys:
            series = np.array(self.values.get(key, []), dtype=float)
            if series.size > 0 and not np.all(np.isnan(series)):
                active_keys.append(key)
        if not active_keys:
            return

        n = len(active_keys)
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows), squeeze=False)
        fig.suptitle(page_title, fontsize=14, fontweight="bold")

        for i, key in enumerate(active_keys):
            row, col = divmod(i, cols)
            ax = axes[row][col]
            series = np.array(self.values[key], dtype=float)
            ax.plot(self.steps, series, linewidth=1.2, color=f"C{i}")
            ax.set_title(key, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("global_step", fontsize=8)

        # Hide unused subplots
        for i in range(len(active_keys), rows * cols):
            row, col = divmod(i, cols)
            axes[row][col].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(save_path, dpi=self.dpi)
        plt.close(fig)

    def render(self):
        if plt is None:
            raise RuntimeError("matplotlib is not available.")
        if not self.steps:
            return

        base = self.path.with_suffix("")
        suffix = self.path.suffix or ".png"
        self._render_page(LIVE_PLOT_PAGE1_KEYS, LIVE_PLOT_PAGE1_TITLE, Path(f"{base}_key{suffix}"))
        self._render_page(LIVE_PLOT_PAGE2_KEYS, LIVE_PLOT_PAGE2_TITLE, Path(f"{base}_other{suffix}"))


def _extract_patch_maps(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.detach().float().cpu()
    if logits.dim() == 4:
        if logits.shape[1] == 1:
            return logits[:, 0]
        return logits.mean(dim=1)
    if logits.dim() == 3:
        return logits
    raise ValueError(f"Unexpected PatchGAN logits shape: {tuple(logits.shape)}")


def _safe_file_stem(file_name: str, fallback: str) -> str:
    stem = Path(file_name).stem if file_name else fallback
    cleaned = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in stem).strip("._")
    return cleaned or fallback


def _resolve_sample_image_path(val_dataset, sample_idx: int):
    base_dataset = val_dataset.dataset if isinstance(val_dataset, Subset) else val_dataset
    if not hasattr(base_dataset, "samples"):
        return None

    samples = base_dataset.samples
    if sample_idx < 0 or sample_idx >= len(samples):
        return None

    sample = samples[sample_idx]
    if isinstance(sample, str):
        return sample

    if isinstance(sample, dict):
        image_path = sample.get("image_path", sample.get("path", sample.get("target", "")))
        if image_path and (not os.path.isabs(image_path)):
            base_dir = getattr(base_dataset, "base_dir", "")
            if base_dir:
                image_path = os.path.join(base_dir, image_path)
        return image_path or None

    return None


def _resize_and_center_crop(image: Image.Image, resize_resolution: int, crop_size: int):
    bilinear = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    width, height = image.size
    short_side = min(width, height)
    if short_side <= 0:
        raise ValueError("Invalid image size.")

    scale = float(resize_resolution) / float(short_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = image.resize((new_width, new_height), resample=bilinear)

    left = max(0, (new_width - crop_size) // 2)
    top = max(0, (new_height - crop_size) // 2)
    return image.crop((left, top, left + crop_size, top + crop_size))


def _load_aligned_image(
    val_dataset,
    sample_idx: int,
    eval_resize_resolution: int,
    eval_crop_size: int,
):
    image_path = _resolve_sample_image_path(val_dataset, sample_idx)
    if image_path is None:
        return None

    try:
        crop_size = int(eval_crop_size) if eval_crop_size and eval_crop_size > 0 else int(eval_resize_resolution)
        image = Image.open(image_path).convert("RGB")
        image = _resize_and_center_crop(image, int(eval_resize_resolution), crop_size)
        return np.asarray(image, dtype=np.uint8)
    except Exception:
        return None


def _tensor_image_to_uint8(image_tensor: torch.Tensor):
    if not torch.is_tensor(image_tensor):
        return None
    image = image_tensor.detach().float().cpu().clamp(0, 1)
    if image.dim() != 3:
        return None
    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0).contiguous()
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 3)
    if image.shape[-1] != 3:
        return None
    return (image.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)


def _resize_patch_map(score_map: np.ndarray, out_height: int, out_width: int):
    h, w = score_map.shape
    row_idx = (np.arange(out_height) * h / out_height).astype(int).clip(0, h - 1)
    col_idx = (np.arange(out_width) * w / out_width).astype(int).clip(0, w - 1)
    return score_map[row_idx[:, None], col_idx[None, :]]


def _colorize_score_map(score_map: np.ndarray):
    score_map = np.clip(score_map, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * score_map - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * score_map - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * score_map - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _draw_patch_grid(image_rgb: np.ndarray, grid_h: int, grid_w: int, color):
    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image)
    height, width = image_rgb.shape[:2]

    for col in range(1, grid_w):
        x = int(round(col * width / grid_w))
        draw.line([(x, 0), (x, height)], fill=color, width=1)
    for row in range(1, grid_h):
        y = int(round(row * height / grid_h))
        draw.line([(0, y), (width, y)], fill=color, width=1)
    return np.asarray(image, dtype=np.uint8)


def _make_colorbar(height: int, vmin: float, vmax: float, bar_width: int = 30, total_width: int = 90):
    canvas = np.full((height, total_width, 3), 255, dtype=np.uint8)
    margin = 10
    bar_height = max(1, height - 2 * margin)

    gradient = np.linspace(1.0, 0.0, bar_height).reshape(-1, 1)
    gradient = np.repeat(gradient, bar_width, axis=1)
    colored = _colorize_score_map(gradient)
    canvas[margin:margin + bar_height, 5:5 + bar_width] = colored

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    text_x = 5 + bar_width + 3
    draw.text((text_x, margin - 2), f"{vmax:.3f}", fill=(0, 0, 0))
    draw.text((text_x, margin + bar_height - 10), f"{vmin:.3f}", fill=(0, 0, 0))
    return np.asarray(image, dtype=np.uint8)


def _build_patch_score_panel(base_image: np.ndarray, score_map: np.ndarray, min_cell_px: int = 32):
    score_map = np.asarray(score_map, dtype=np.float32)
    grid_h, grid_w = score_map.shape
    height, width = base_image.shape[:2]

    vmin, vmax = float(score_map.min()), float(score_map.max())
    if vmax - vmin < 1e-6:
        normalized = np.full_like(score_map, 0.5)
    else:
        normalized = (score_map - vmin) / (vmax - vmin)

    upsampled = _resize_patch_map(normalized, height, width)
    heatmap = _colorize_score_map(upsampled)
    overlay = (
        0.55 * base_image.astype(np.float32) + 0.45 * heatmap.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    cell_size = min(width / grid_w, height / grid_h)
    if cell_size >= float(min_cell_px):
        original_grid = _draw_patch_grid(base_image.copy(), grid_h, grid_w, color=(180, 180, 180))
        heatmap_grid = _draw_patch_grid(heatmap, grid_h, grid_w, color=(255, 255, 255))
        overlay_grid = _draw_patch_grid(overlay, grid_h, grid_w, color=(255, 255, 255))
    else:
        original_grid = base_image.copy()
        heatmap_grid = heatmap
        overlay_grid = overlay

    colorbar = _make_colorbar(height, vmin, vmax)
    return np.concatenate([original_grid, heatmap_grid, overlay_grid, colorbar], axis=1)


def _save_patch_score_visualizations(
    ordered_records: list,
    output_dir: Path,
    val_dataset,
    eval_resize_resolution: int,
    eval_crop_size: int,
    vis_max_samples: int,
    selected_sample_indices: list,
    recon_image_by_sample_idx: dict,
    min_cell_px: int,
):
    if vis_max_samples is not None and vis_max_samples <= 0:
        return
    if len(ordered_records) == 0:
        return

    real_dir = output_dir / "patch_vis" / "real"
    recon_dir = output_dir / "patch_vis" / "recon"
    real_dir.mkdir(exist_ok=True, parents=True)
    recon_dir.mkdir(exist_ok=True, parents=True)

    selected_records = ordered_records
    if selected_sample_indices is not None:
        sample_idx_to_record = {int(record["sample_idx"]): record for record in ordered_records}
        selected_records = [
            sample_idx_to_record[sample_idx]
            for sample_idx in selected_sample_indices
            if sample_idx in sample_idx_to_record
        ]

    max_samples = len(selected_records) if vis_max_samples is None else min(len(selected_records), vis_max_samples)
    selected_records = selected_records[:max_samples]
    crop_size = (
        int(eval_crop_size)
        if eval_crop_size is not None and int(eval_crop_size) > 0
        else int(eval_resize_resolution)
    )

    for row_id, record in enumerate(selected_records):
        sample_idx = int(record["sample_idx"])
        file_name = record.get("file_name", "")
        fallback_name = f"sample_{sample_idx:08d}"
        stem = _safe_file_stem(file_name, fallback_name)
        image_name = f"{row_id:05d}_{sample_idx:08d}_{stem}.png"

        real_base_image = (
            _load_aligned_image(val_dataset, sample_idx, eval_resize_resolution, crop_size)
            if val_dataset is not None
            else None
        )
        if real_base_image is None:
            real_base_image = np.full((crop_size, crop_size, 3), 127, dtype=np.uint8)

        recon_base_image = recon_image_by_sample_idx.get(sample_idx) if recon_image_by_sample_idx is not None else None
        if recon_base_image is None:
            recon_base_image = real_base_image

        real_panel = _build_patch_score_panel(
            real_base_image, record["real_sigmoid"], min_cell_px=min_cell_px
        )
        recon_panel = _build_patch_score_panel(
            recon_base_image, record["recon_sigmoid"], min_cell_px=min_cell_px
        )

        Image.fromarray(real_panel).save(real_dir / image_name)
        Image.fromarray(recon_panel).save(recon_dir / image_name)


def save_patch_scores(
    patch_records: list,
    output_dir: Path,
    val_dataset=None,
    eval_resize_resolution: int = 1024,
    eval_crop_size: int = 1024,
    vis_max_samples: int = None,
    selected_sample_indices: list = None,
    recon_image_by_sample_idx: dict = None,
    save_summary: bool = True,
    save_maps: bool = True,
    min_cell_px: int = 32,
):
    if len(patch_records) == 0:
        return

    output_dir.mkdir(exist_ok=True, parents=True)
    ordered_records = sorted(patch_records, key=lambda x: int(x["sample_idx"]))

    if save_summary:
        summary_path = output_dir / "summary.csv"
        fieldnames = [
            "sample_idx", "file_name", "row_id",
            "real_logits_mean", "real_logits_std", "real_logits_min", "real_logits_max",
            "recon_logits_mean", "recon_logits_std", "recon_logits_min", "recon_logits_max",
            "real_sigmoid_mean", "real_sigmoid_std", "real_sigmoid_min", "real_sigmoid_max",
            "recon_sigmoid_mean", "recon_sigmoid_std", "recon_sigmoid_min", "recon_sigmoid_max",
        ]
        with open(summary_path, "w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
            writer.writeheader()
            for row_id, record in enumerate(ordered_records):
                real_logits = np.asarray(record["real_logits"], dtype=np.float32)
                recon_logits = np.asarray(record["recon_logits"], dtype=np.float32)
                real_sigmoid = np.asarray(record["real_sigmoid"], dtype=np.float32)
                recon_sigmoid = np.asarray(record["recon_sigmoid"], dtype=np.float32)

                writer.writerow(
                    {
                        "sample_idx": int(record["sample_idx"]),
                        "file_name": record.get("file_name", ""),
                        "row_id": row_id,
                        "real_logits_mean": float(real_logits.mean()),
                        "real_logits_std": float(real_logits.std()),
                        "real_logits_min": float(real_logits.min()),
                        "real_logits_max": float(real_logits.max()),
                        "recon_logits_mean": float(recon_logits.mean()),
                        "recon_logits_std": float(recon_logits.std()),
                        "recon_logits_min": float(recon_logits.min()),
                        "recon_logits_max": float(recon_logits.max()),
                        "real_sigmoid_mean": float(real_sigmoid.mean()),
                        "real_sigmoid_std": float(real_sigmoid.std()),
                        "real_sigmoid_min": float(real_sigmoid.min()),
                        "real_sigmoid_max": float(real_sigmoid.max()),
                        "recon_sigmoid_mean": float(recon_sigmoid.mean()),
                        "recon_sigmoid_std": float(recon_sigmoid.std()),
                        "recon_sigmoid_min": float(recon_sigmoid.min()),
                        "recon_sigmoid_max": float(recon_sigmoid.max()),
                    }
                )

    if save_maps:
        _save_patch_score_visualizations(
            ordered_records=ordered_records,
            output_dir=output_dir,
            val_dataset=val_dataset,
            eval_resize_resolution=int(eval_resize_resolution),
            eval_crop_size=int(eval_crop_size) if eval_crop_size is not None else int(eval_resize_resolution),
            vis_max_samples=vis_max_samples,
            selected_sample_indices=selected_sample_indices,
            recon_image_by_sample_idx=recon_image_by_sample_idx,
            min_cell_px=min_cell_px,
        )


def log_patch_scores_to_wandb(patch_records: list, step: int, prefix: str, enabled: bool = True):
    if (not enabled) or len(patch_records) == 0:
        return

    ordered_records = sorted(patch_records, key=lambda x: int(x["sample_idx"]))
    real_logits_flat = np.concatenate(
        [np.asarray(record["real_logits"], dtype=np.float32).reshape(-1) for record in ordered_records]
    )
    recon_logits_flat = np.concatenate(
        [np.asarray(record["recon_logits"], dtype=np.float32).reshape(-1) for record in ordered_records]
    )
    real_sigmoid_flat = np.concatenate(
        [np.asarray(record["real_sigmoid"], dtype=np.float32).reshape(-1) for record in ordered_records]
    )
    recon_sigmoid_flat = np.concatenate(
        [np.asarray(record["recon_sigmoid"], dtype=np.float32).reshape(-1) for record in ordered_records]
    )

    wandb.log(
        {
            f"{prefix}/patch_real_logits_mean": float(real_logits_flat.mean()),
            f"{prefix}/patch_recon_logits_mean": float(recon_logits_flat.mean()),
            f"{prefix}/patch_real_sigmoid_mean": float(real_sigmoid_flat.mean()),
            f"{prefix}/patch_recon_sigmoid_mean": float(recon_sigmoid_flat.mean()),
            f"{prefix}/patch_real_logits_hist": wandb.Histogram(real_logits_flat),
            f"{prefix}/patch_recon_logits_hist": wandb.Histogram(recon_logits_flat),
            f"{prefix}/patch_real_sigmoid_hist": wandb.Histogram(real_sigmoid_flat),
            f"{prefix}/patch_recon_sigmoid_hist": wandb.Histogram(recon_sigmoid_flat),
        },
        step=step,
    )


def valid(global_rank, rank, model, discriminator, val_dataloader, precision, args):
    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation")

    psnr_list = []
    ssim_list = []
    lpips_list = []
    image_log = []
    patch_records = []
    orig_image_records = []
    recon_image_records = []
    logged_sample_indices = []
    num_image_log = args.eval_num_image_log
    if args.enable_val_image_dump and args.val_image_dump_max_samples > 0:
        num_image_log = max(num_image_log, args.val_image_dump_max_samples)
    if args.enable_patch_score_vis:
        patch_image_budget = (
            args.patch_score_vis_max_samples
            if args.patch_score_vis_max_samples > 0
            else args.eval_num_image_log
        )
        num_image_log = max(num_image_log, patch_image_budget)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs = batch["image"].to(rank, non_blocking=True)
            sample_indices = batch.get("index")
            if sample_indices is None:
                start_idx = batch_idx * inputs.shape[0]
                sample_indices = list(range(start_idx, start_idx + inputs.shape[0]))
            elif torch.is_tensor(sample_indices):
                sample_indices = sample_indices.detach().cpu().tolist()
            else:
                sample_indices = [int(i) for i in sample_indices]

            file_names = batch.get("file_name")
            if file_names is None:
                file_names = [f"sample_{sample_indices[i]}" for i in range(inputs.shape[0])]
            else:
                file_names = [str(name) for name in file_names]

            need_patch_scores = args.enable_patch_score_vis and discriminator is not None
            with torch.amp.autocast("cuda", dtype=precision):
                outputs = model(inputs)
                recon = outputs.sample
                logits_real = discriminator(inputs) if need_patch_scores else None
                logits_recon = discriminator(recon) if need_patch_scores else None

            inputs_01 = ((inputs.float().clamp(-1, 1) + 1) / 2).contiguous()
            recon_01 = ((recon.float().clamp(-1, 1) + 1) / 2).contiguous()

            psnr = calculate_psnr(recon_01, inputs_01)
            ssim = np.mean(
                list(
                    calculate_ssim(recon_01, inputs_01)["value"].values()
                )
            )
            lpips_score = calculate_lpips(
                recon_01, inputs_01, inputs_01.device
            )

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips_score)

            if need_patch_scores:
                real_logits_maps = _extract_patch_maps(logits_real)
                recon_logits_maps = _extract_patch_maps(logits_recon)
                real_sigmoid_maps = torch.sigmoid(real_logits_maps)
                recon_sigmoid_maps = torch.sigmoid(recon_logits_maps)

                for idx in range(recon_01.shape[0]):
                    sample_idx = int(sample_indices[idx])
                    file_name = file_names[idx] if idx < len(file_names) else f"sample_{sample_idx}"
                    patch_records.append(
                        {
                            "sample_idx": sample_idx,
                            "file_name": file_name,
                            "real_logits": real_logits_maps[idx].numpy(),
                            "recon_logits": recon_logits_maps[idx].numpy(),
                            "real_sigmoid": real_sigmoid_maps[idx].numpy(),
                            "recon_sigmoid": recon_sigmoid_maps[idx].numpy(),
                        }
                    )

            for idx in range(recon_01.shape[0]):
                if num_image_log <= 0:
                    break
                sample_idx = int(sample_indices[idx])
                image_log.append(recon_01[idx].detach().cpu().numpy())
                orig_image_records.append(
                    {
                        "sample_idx": sample_idx,
                        "orig_image": _tensor_image_to_uint8(inputs_01[idx]),
                    }
                )
                recon_image_records.append(
                    {
                        "sample_idx": sample_idx,
                        "recon_image": _tensor_image_to_uint8(recon_01[idx]),
                    }
                )
                logged_sample_indices.append(sample_idx)
                num_image_log -= 1

            if global_rank == 0 and bar is not None:
                bar.update()

    return (
        psnr_list,
        ssim_list,
        lpips_list,
        image_log,
        patch_records,
        orig_image_records,
        recon_image_records,
        logged_sample_indices,
    )


def gather_valid_result(
    psnr_list,
    ssim_list,
    lpips_list,
    image_log_list,
    patch_records,
    orig_image_records,
    recon_image_records,
    logged_sample_indices,
    world_size,
    collect_patch_records: bool = True,
):
    gathered_psnr_list = [None for _ in range(world_size)]
    gathered_ssim_list = [None for _ in range(world_size)]
    gathered_lpips_list = [None for _ in range(world_size)]
    gathered_image_logs = [None for _ in range(world_size)]
    gathered_patch_records = [None for _ in range(world_size)] if collect_patch_records else None
    gathered_orig_image_records = [None for _ in range(world_size)]
    gathered_recon_image_records = [None for _ in range(world_size)]
    gathered_logged_sample_indices = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_psnr_list, psnr_list)
    dist.all_gather_object(gathered_ssim_list, ssim_list)
    dist.all_gather_object(gathered_lpips_list, lpips_list)
    dist.all_gather_object(gathered_image_logs, image_log_list)
    if collect_patch_records:
        dist.all_gather_object(gathered_patch_records, patch_records)
    dist.all_gather_object(gathered_orig_image_records, orig_image_records)
    dist.all_gather_object(gathered_recon_image_records, recon_image_records)
    dist.all_gather_object(gathered_logged_sample_indices, logged_sample_indices)

    deduplicated_patch_records = []
    if collect_patch_records:
        all_patch_records = list(chain(*gathered_patch_records))
        patch_record_dict = {}
        for record in all_patch_records:
            sample_idx = int(record["sample_idx"])
            if sample_idx not in patch_record_dict:
                patch_record_dict[sample_idx] = record
        deduplicated_patch_records = [patch_record_dict[idx] for idx in sorted(patch_record_dict.keys())]

    orig_image_by_sample_idx = {}
    all_orig_records = list(chain(*gathered_orig_image_records))
    for record in all_orig_records:
        sample_idx = int(record.get("sample_idx", -1))
        orig_image = record.get("orig_image")
        if sample_idx >= 0 and orig_image is not None and sample_idx not in orig_image_by_sample_idx:
            orig_image_by_sample_idx[sample_idx] = orig_image

    recon_image_by_sample_idx = {}
    all_recon_records = list(chain(*gathered_recon_image_records))
    for record in all_recon_records:
        sample_idx = int(record.get("sample_idx", -1))
        recon_image = record.get("recon_image")
        if sample_idx >= 0 and recon_image is not None and sample_idx not in recon_image_by_sample_idx:
            recon_image_by_sample_idx[sample_idx] = recon_image

    dedup_logged_sample_indices = []
    for sample_idx in chain(*gathered_logged_sample_indices):
        sample_idx = int(sample_idx)
        if sample_idx not in dedup_logged_sample_indices:
            dedup_logged_sample_indices.append(sample_idx)

    return (
        np.array(gathered_psnr_list).mean(),
        np.array(gathered_ssim_list).mean(),
        np.array(gathered_lpips_list).mean(),
        list(chain(*gathered_image_logs)),
        deduplicated_patch_records,
        orig_image_by_sample_idx,
        recon_image_by_sample_idx,
        dedup_logged_sample_indices,
    )


def train(args):
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger = setup_logger(rank)

    if args.csv_log_steps <= 0:
        raise ValueError("`--csv_log_steps` must be >= 1.")
    if args.live_plot_every_steps <= 0:
        raise ValueError("`--live_plot_every_steps` must be >= 1.")
    if args.live_plot_history < 0:
        raise ValueError("`--live_plot_history` must be >= 0.")
    if args.live_plot_dpi <= 0:
        raise ValueError("`--live_plot_dpi` must be >= 1.")
    if args.patch_score_vis_max_samples < 0:
        raise ValueError("`--patch_score_vis_max_samples` must be >= 0.")
    if args.patch_score_min_cell_px < 0:
        raise ValueError("`--patch_score_min_cell_px` must be >= 0.")
    if args.val_image_dump_max_samples < 0:
        raise ValueError("`--val_image_dump_max_samples` must be >= 0.")
    if args.dataloader_prefetch_factor <= 0:
        raise ValueError("`--dataloader_prefetch_factor` must be >= 1.")

    ckpt_dir = Path(args.ckpt_dir) / Path(get_exp_name(args))
    if global_rank == 0:
        ckpt_dir.mkdir(exist_ok=True, parents=True)
    dist.barrier()

    model_cls = ModelRegistry.get_model(args.model_name)
    if not model_cls:
        raise ModuleNotFoundError(
            f"`{args.model_name}` not in {str(ModelRegistry._models.keys())}."
        )

    if args.pretrained_model_name_or_path is not None:
        logger.warning(
            f"Loading pretrained model from `{args.pretrained_model_name_or_path}`."
        )
        model = model_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            low_cpu_mem_usage=False,
            device_map=None,
        )
    else:
        if args.model_config is None:
            raise ValueError("`--model_config` is required when training from scratch.")
        logger.warning("Model will be initialized from config file.")
        model = model_cls.from_config(args.model_config)

    if global_rank == 0:
        model_config = dict(**model.config)
        args_config = dict(**vars(args))
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "wfvae-image"),
            config=dict(**model_config, **args_config),
            name=get_exp_name(args),
        )

    dist.barrier()

    disc_cls = resolve_str_to_obj(args.disc_cls, append=False)
    logger.warning(f"disc_class: {args.disc_cls}")
    disc = disc_cls(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        kl_weight=args.kl_weight,
        logvar_init=args.logvar_init,
        perceptual_weight=args.perceptual_weight,
        loss_type=args.loss_type,
        wavelet_weight=args.wavelet_weight,
    )

    model = model.to(rank)
    model = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=args.find_unused_parameters,
    )
    disc = disc.to(rank)
    disc = DDP(
        disc,
        device_ids=[rank],
        find_unused_parameters=args.find_unused_parameters,
    )

    warmup_dataset_index_cache(
        image_path=args.image_path,
        eval_image_path=args.eval_image_path,
        use_manifest=args.use_manifest,
        global_rank=global_rank,
        logger=logger,
    )

    dataset = TrainImageDataset(
        image_folder=args.image_path,
        resolution=args.resolution,
        cache_file="idx_image.pkl",
        is_main_process=global_rank == 0,
        use_manifest=args.use_manifest,
        manifest_path=args.image_path if args.use_manifest else None,
    )
    ddp_sampler = CustomDistributedSampler(dataset)
    train_num_workers = max(0, args.dataset_num_worker)
    train_persistent_workers = train_num_workers > 0 and (not args.disable_persistent_workers)
    train_loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=ddp_sampler,
        pin_memory=True,
        num_workers=train_num_workers,
        drop_last=True,
        persistent_workers=train_persistent_workers,
    )
    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
    dataloader = DataLoader(**train_loader_kwargs)

    val_dataset = ValidImageDataset(
        image_dir=args.eval_image_path,
        resolution=args.eval_resolution,
        crop_size=args.eval_crop_size,
        cache_file="idx_image_eval.pkl",
        is_main_process=global_rank == 0,
        use_manifest=args.use_manifest,
        manifest_path=args.eval_image_path if args.use_manifest else None,
    )
    if args.eval_subset_size and args.eval_subset_size > 0:
        indices = list(range(min(args.eval_subset_size, len(val_dataset))))
        val_dataset = Subset(val_dataset, indices=indices)
    val_sampler = CustomDistributedSampler(val_dataset)
    val_num_workers = max(1, args.dataset_num_worker // 2)
    val_persistent_workers = val_num_workers > 0 and (not args.disable_persistent_workers)
    val_loader_kwargs = dict(
        dataset=val_dataset,
        batch_size=args.eval_batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=val_num_workers,
        persistent_workers=val_persistent_workers,
    )
    if val_num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
    val_dataloader = DataLoader(**val_loader_kwargs)

    modules_to_train = [module for module in model.module.get_decoder()]
    if args.freeze_encoder:
        for module in model.module.get_encoder():
            module.eval()
            module.requires_grad_(False)
        logger.info("Encoder is frozen.")
    else:
        modules_to_train += [module for module in model.module.get_encoder()]

    parameters_to_train = []
    for module in modules_to_train:
        parameters_to_train.extend(
            list(filter(lambda p: p.requires_grad, module.parameters()))
        )

    gen_optimizer = torch.optim.AdamW(
        parameters_to_train,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    disc_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda")
    precision = torch.bfloat16
    if args.mix_precision == "fp16":
        precision = torch.float16
    elif args.mix_precision == "fp32":
        precision = torch.float32

    start_epoch = 0
    current_step = 0
    checkpoint = None
    if args.resume_from_checkpoint:
        if not os.path.isfile(args.resume_from_checkpoint):
            raise ValueError(f"`{args.resume_from_checkpoint}` is not a valid checkpoint.")

        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.module.load_state_dict(checkpoint["state_dict"]["gen_model"], strict=False)

        if not args.not_resume_optimizer:
            gen_optimizer.load_state_dict(checkpoint["optimizer_state"]["gen_optimizer"])

        if not args.not_resume_discriminator:
            disc_state = checkpoint["state_dict"].get(
                "disc_model", checkpoint["state_dict"].get("dics_model")
            )
            if disc_state is None:
                raise KeyError("Cannot find discriminator state in checkpoint.")
            disc.module.load_state_dict(disc_state)
            disc_optimizer.load_state_dict(checkpoint["optimizer_state"]["disc_optimizer"])
            scaler.load_state_dict(checkpoint["scaler_state"])

        ddp_sampler.load_state_dict(checkpoint["sampler_state"])
        start_epoch = checkpoint["sampler_state"]["epoch"]
        current_step = checkpoint["current_step"]

        logger.info(
            f"Checkpoint loaded from {args.resume_from_checkpoint}, "
            f"start epoch {start_epoch}, step {current_step}."
        )

    ema = None
    if args.ema:
        logger.warning(f"EMA enabled with decay={args.ema_decay}.")
        ema = EMA(model, args.ema_decay)
        ema.register()
        if checkpoint is not None:
            loaded_ema_state = checkpoint.get("ema_state_dict", {})
            if isinstance(loaded_ema_state, dict) and len(loaded_ema_state) > 0:
                restored = 0
                missing = 0
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    ema_value = loaded_ema_state.get(name)
                    if ema_value is None and name.startswith("module."):
                        ema_value = loaded_ema_state.get(name[len("module."):])
                    if ema_value is None and (not name.startswith("module.")):
                        ema_value = loaded_ema_state.get(f"module.{name}")
                    if torch.is_tensor(ema_value):
                        ema.shadow[name] = ema_value.to(
                            device=param.device, dtype=param.dtype
                        ).clone()
                        restored += 1
                    else:
                        missing += 1
                logger.info(
                    f"EMA state restored from checkpoint: restored={restored}, missing={missing}."
                )
            else:
                logger.warning(
                    "Checkpoint has no valid `ema_state_dict`; EMA shadow will be initialized "
                    "from current model parameters."
                )

    num_micro_batches_per_epoch = len(dataloader)
    num_update_steps_per_epoch = math.ceil(num_micro_batches_per_epoch / args.grad_accum_steps)
    max_steps = args.max_steps or (args.epochs * num_update_steps_per_epoch)

    csv_logger = None
    if global_rank == 0 and not args.disable_csv_log:
        csv_path = Path(args.csv_log_path) if args.csv_log_path else (ckpt_dir / "training_losses.csv")
        csv_logger = TrainingCSVLogger(csv_path)
        logger.info(f"CSV logging enabled: `{csv_path}` (every {args.csv_log_steps} update step(s)).")
    elif global_rank == 0 and args.disable_csv_log:
        logger.info("CSV logging is disabled.")

    live_plotter = None
    if global_rank == 0 and args.enable_live_plot:
        if plt is None:
            logger.warning("Live loss plot is disabled because matplotlib is not available.")
        else:
            live_plot_path = (
                Path(args.live_plot_path) if args.live_plot_path else (ckpt_dir / "training_losses_live.png")
            )
            live_plotter = LiveLossPlotter(
                live_plot_path,
                history=args.live_plot_history,
                dpi=args.live_plot_dpi,
            )
            if csv_logger is not None:
                live_plotter.load_history_from_csv(csv_logger.path)
                if live_plotter.steps:
                    logger.info(f"Live plotter restored {len(live_plotter.steps)} history points from CSV.")
            logger.info(
                f"Live loss plot enabled: `{live_plot_path}` "
                f"(every {args.live_plot_every_steps} update step(s))."
            )
    elif global_rank == 0 and not args.enable_live_plot:
        logger.info("Live loss plot is disabled.")
    live_plot_failed = False

    if global_rank == 0:
        logger.info(f"Generator: {total_params(model.module)}M")
        logger.info(f"  Encoder: {total_params(model.module.encoder)}M")
        logger.info(f"  Decoder: {total_params(model.module.decoder)}M")
        logger.info(f"Discriminator: {total_params(disc.module)}M")
        logger.info(f"Mixed precision: {args.mix_precision}")
        logger.info(f"Gradient accumulation: {args.grad_accum_steps}")
        logger.info(f"Train samples: {len(dataset)}")
        logger.info(
            "DataLoader: "
            f"train_workers={train_num_workers}, val_workers={val_num_workers}, "
            f"train_persistent_workers={train_persistent_workers}, "
            f"val_persistent_workers={val_persistent_workers}, "
            f"prefetch_factor={args.dataloader_prefetch_factor}"
        )
        logger.info(
            f"Global batch size per update: {args.batch_size} x {world_size} x {args.grad_accum_steps}"
        )
        if args.enable_patch_score_vis:
            vis_max_samples = (
                args.patch_score_vis_max_samples
                if args.patch_score_vis_max_samples > 0
                else args.eval_num_image_log
            )
            logger.info(
                f"PatchGAN score export enabled: vis_max_samples={vis_max_samples}, "
                f"summary={args.patch_score_save_summary}, maps={args.patch_score_save_maps}, "
                f"wandb={args.patch_score_log_wandb}"
            )
        else:
            logger.info("PatchGAN score export is disabled.")
        if args.enable_val_image_dump:
            dump_max_samples = (
                args.val_image_dump_max_samples
                if args.val_image_dump_max_samples > 0
                else args.eval_num_image_log
            )
            logger.info(f"Validation image dump enabled: max_samples={dump_max_samples}")
        else:
            logger.info("Validation image dump is disabled.")

    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=max_steps, desc="Train")
        bar.update(current_step)

    def log_validation(model_to_eval, epoch_idx, batch_idx, name=""):
        set_eval(modules_to_train)
        discriminator = None
        was_disc_training = None
        if args.enable_patch_score_vis:
            discriminator = disc.module.discriminator
            was_disc_training = discriminator.training
            discriminator.eval()
        try:
            (
                psnr_list,
                ssim_list,
                lpips_list,
                image_log,
                patch_records,
                orig_image_records,
                recon_image_records,
                logged_sample_indices,
            ) = valid(
                global_rank,
                rank,
                model_to_eval,
                discriminator,
                val_dataloader,
                precision,
                args,
            )
            (
                valid_psnr,
                valid_ssim,
                valid_lpips,
                valid_image_log,
                valid_patch_records,
                orig_image_by_sample_idx,
                recon_image_by_sample_idx,
                valid_logged_sample_indices,
            ) = gather_valid_result(
                psnr_list,
                ssim_list,
                lpips_list,
                image_log,
                patch_records,
                orig_image_records,
                recon_image_records,
                logged_sample_indices,
                world_size,
                collect_patch_records=args.enable_patch_score_vis,
            )
        finally:
            if discriminator is not None:
                discriminator.train(was_disc_training)

        if global_rank == 0:
            suffix = f"_{name}" if name else ""
            wandb_images = []
            for idx, image in enumerate(valid_image_log[: args.eval_num_image_log]):
                wandb_images.append(
                    wandb.Image(np.transpose(image, (1, 2, 0)), caption=f"{name or 'main'}_{idx}")
                )
            if wandb_images:
                wandb.log({f"val{suffix}/recon": wandb_images}, step=current_step)
            wandb.log(
                {
                    f"val{suffix}/psnr": valid_psnr,
                    f"val{suffix}/ssim": valid_ssim,
                    f"val{suffix}/lpips": valid_lpips,
                },
                step=current_step,
            )
            if csv_logger is not None:
                csv_logger.log(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "epoch": epoch_idx,
                        "batch_idx": batch_idx,
                        "global_step": current_step,
                        "phase": "val",
                        "is_ema": 1 if name == "ema" else 0,
                        "val_psnr": to_scalar(valid_psnr),
                        "val_ssim": to_scalar(valid_ssim),
                        "val_lpips": to_scalar(valid_lpips),
                    }
                )

            # Compute unified sample indices for both patch scores and val images
            unified_max_samples = max(
                args.patch_score_vis_max_samples if args.enable_patch_score_vis and args.patch_score_vis_max_samples > 0 else 0,
                args.val_image_dump_max_samples if args.enable_val_image_dump and args.val_image_dump_max_samples > 0 else 0,
                args.eval_num_image_log,
            )
            # Prefer logged sample indices (processing order); fall back to common keys
            unified_sample_indices = valid_logged_sample_indices[:unified_max_samples]
            if not unified_sample_indices:
                common_indices = sorted(
                    set(orig_image_by_sample_idx.keys()) & set(recon_image_by_sample_idx.keys())
                )
                unified_sample_indices = common_indices[:unified_max_samples]

            if args.enable_patch_score_vis and len(valid_patch_records) > 0:
                try:
                    patch_score_dir = ckpt_dir / "val_patch_scores" / f"step_{current_step:08d}{suffix}"
                    save_patch_scores(
                        valid_patch_records,
                        patch_score_dir,
                        val_dataset=val_dataset,
                        eval_resize_resolution=args.eval_resolution,
                        eval_crop_size=args.eval_crop_size,
                        vis_max_samples=len(unified_sample_indices),
                        selected_sample_indices=unified_sample_indices if unified_sample_indices else None,
                        recon_image_by_sample_idx=recon_image_by_sample_idx,
                        save_summary=args.patch_score_save_summary,
                        save_maps=args.patch_score_save_maps,
                        min_cell_px=args.patch_score_min_cell_px,
                    )
                    log_patch_scores_to_wandb(
                        valid_patch_records,
                        current_step,
                        f"val{suffix}",
                        enabled=args.patch_score_log_wandb,
                    )
                    logger.info(
                        f"PatchGAN patch scores saved to `{patch_score_dir}` "
                        f"(summary={args.patch_score_save_summary}, maps={args.patch_score_save_maps})."
                    )
                except Exception as exc:
                    logger.warning(f"PatchGAN patch score export failed at step {current_step}: {exc}")

            if args.enable_val_image_dump and unified_sample_indices:
                try:
                    orig_dir = ckpt_dir / "val_images" / "original"
                    recon_dir = ckpt_dir / "val_images" / "reconstructed"
                    orig_dir.mkdir(exist_ok=True, parents=True)
                    recon_dir.mkdir(exist_ok=True, parents=True)

                    num_saved = 0
                    for idx, sample_idx in enumerate(unified_sample_indices):
                        orig_img = orig_image_by_sample_idx.get(sample_idx)
                        recon_img = recon_image_by_sample_idx.get(sample_idx)
                        if orig_img is None or recon_img is None:
                            continue
                        Image.fromarray(orig_img).save(
                            orig_dir / f"step_{current_step}_original{suffix}_{idx:03d}_sid{sample_idx}.png"
                        )
                        Image.fromarray(recon_img).save(
                            recon_dir / f"step_{current_step}_recon{suffix}_{idx:03d}_sid{sample_idx}.png"
                        )
                        num_saved += 1
                    logger.info(f"Validation images saved: {num_saved} pair(s) with suffix `{suffix or '_main'}`.")
                except Exception as exc:
                    logger.warning(f"Validation image dump failed at step {current_step}: {exc}")
        return valid_psnr, valid_ssim, valid_lpips

    dist.barrier()
    try:
        for epoch in range(start_epoch, args.epochs):
            if current_step >= max_steps:
                break

            set_train(modules_to_train)
            ddp_sampler.set_epoch(epoch)

            epoch_start_sample_index = int(getattr(ddp_sampler, "current_index", 0))
            epoch_remaining_samples = max(0, len(ddp_sampler) - epoch_start_sample_index)
            epoch_remaining_micro_batches = epoch_remaining_samples // args.batch_size

            gen_optimizer.zero_grad(set_to_none=True)
            disc_optimizer.zero_grad(set_to_none=True)

            micro_batches_in_gen_window = 0
            micro_batches_in_disc_window = 0
            gen_accum_window = 0
            disc_accum_window = 0

            # Retain last known logs for the phase not active in this step.
            last_g_loss = None
            last_g_log = {}
            last_d_loss = None
            last_d_log = {}
            last_posterior = None
            _prev_step_gen = None

            for batch_idx, batch in enumerate(dataloader):
                inputs = batch["image"].to(rank, non_blocking=True)

                # Alternating training: even steps -> Generator, odd steps -> Discriminator
                if (
                    current_step % 2 == 1
                    and current_step >= disc.module.discriminator_iter_start
                ):
                    step_gen = False
                    step_dis = True
                else:
                    step_gen = True
                    step_dis = False

                if step_gen != _prev_step_gen:
                    set_modules_requires_grad(modules_to_train, step_gen)
                    _prev_step_gen = step_gen

                # Forward pass: D-step does not require autograd graph for VAE.
                if step_gen:
                    with torch.amp.autocast("cuda", dtype=precision):
                        outputs = model(inputs)
                else:
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", dtype=precision):
                            outputs = model(inputs)

                recon = outputs.sample
                posterior = outputs.latent_dist
                wavelet_coeffs = outputs.extra_output if (args.wavelet_loss and step_gen) else None

                # Generator step
                if step_gen:
                    if micro_batches_in_gen_window == 0:
                        micro_batches_left = max(1, epoch_remaining_micro_batches - batch_idx)
                        gen_accum_window = min(args.grad_accum_steps, micro_batches_left)
                    micro_batches_in_gen_window += 1
                    should_step = micro_batches_in_gen_window >= gen_accum_window
                    sync_context = nullcontext() if should_step else model.no_sync()

                    with sync_context:
                        with torch.amp.autocast("cuda", dtype=precision):
                            g_loss, g_log = disc(
                                inputs,
                                recon,
                                posterior,
                                optimizer_idx=0,
                                global_step=current_step,
                                last_layer=model.module.get_last_layer(),
                                wavelet_coeffs=wavelet_coeffs,
                                split="train",
                            )
                        scaler.scale(g_loss / gen_accum_window).backward()

                    if not should_step:
                        continue

                    if args.clip_grad_norm > 0:
                        scaler.unscale_(gen_optimizer)
                        clip_grad_norm_(parameters_to_train, args.clip_grad_norm)

                    scaler.step(gen_optimizer)
                    scaler.update()
                    gen_optimizer.zero_grad(set_to_none=True)
                    micro_batches_in_gen_window = 0

                    if ema is not None:
                        ema.update()

                    last_g_loss = g_loss
                    last_g_log = g_log
                    last_posterior = posterior

                # Discriminator step
                if step_dis:
                    if micro_batches_in_disc_window == 0:
                        micro_batches_left = max(1, epoch_remaining_micro_batches - batch_idx)
                        disc_accum_window = min(args.grad_accum_steps, micro_batches_left)
                    micro_batches_in_disc_window += 1
                    should_step = micro_batches_in_disc_window >= disc_accum_window
                    sync_context = nullcontext() if should_step else disc.no_sync()

                    with sync_context:
                        with torch.amp.autocast("cuda", dtype=precision):
                            d_loss, d_log = disc(
                                inputs,
                                recon.detach(),
                                posterior,
                                optimizer_idx=1,
                                global_step=current_step,
                                last_layer=None,
                                split="train",
                            )
                        scaler.scale(d_loss / disc_accum_window).backward()

                    if not should_step:
                        continue

                    if args.clip_grad_norm > 0:
                        scaler.unscale_(disc_optimizer)
                        clip_grad_norm_(disc.module.discriminator.parameters(), args.clip_grad_norm)

                    scaler.step(disc_optimizer)
                    scaler.update()
                    disc_optimizer.zero_grad(set_to_none=True)
                    micro_batches_in_disc_window = 0

                    last_d_loss = d_loss
                    last_d_log = d_log
                    last_posterior = posterior

                current_step += 1
                if global_rank == 0:
                    bar.update(1)

                # Use current phase values, fall back to last known values
                cur_g_loss = g_loss if step_gen else last_g_loss
                cur_g_log = g_log if step_gen else last_g_log
                cur_d_loss = d_loss if step_dis else last_d_loss
                cur_d_log = d_log if step_dis else last_d_log
                cur_posterior = last_posterior if last_posterior is not None else posterior

                need_csv_log = (
                    global_rank == 0
                    and csv_logger is not None
                    and current_step % args.csv_log_steps == 0
                )
                need_live_plot = (
                    global_rank == 0
                    and live_plotter is not None
                    and not live_plot_failed
                    and current_step % args.live_plot_every_steps == 0
                )
                need_wandb_log = global_rank == 0 and current_step % args.log_steps == 0
                need_latents_std = need_csv_log or need_live_plot or need_wandb_log

                latents_std = None
                if need_latents_std:
                    latents_std = (
                        cur_posterior.sample().std().item()
                        if cur_posterior is not None
                        else 0.0
                    )

                train_metrics = None
                if need_csv_log or need_live_plot:
                    train_metrics = {
                        "train_total_loss": to_scalar(cur_g_log.get("train/total_loss")),
                        "train_logvar": to_scalar(cur_g_log.get("train/logvar")),
                        "train_kl_loss": to_scalar(cur_g_log.get("train/kl_loss")),
                        "train_nll_loss": to_scalar(cur_g_log.get("train/nll_loss")),
                        "train_rec_loss": to_scalar(cur_g_log.get("train/rec_loss")),
                        "train_wl_loss": to_scalar(cur_g_log.get("train/wl_loss")),
                        "train_distill_loss": to_scalar(cur_g_log.get("train/distill_loss")),
                        "train_d_weight": to_scalar(cur_g_log.get("train/d_weight")),
                        "train_disc_factor": to_scalar(cur_g_log.get("train/disc_factor")),
                        "train_g_loss": to_scalar(cur_g_loss),
                        "train_d_loss": to_scalar(cur_d_loss),
                        "train_disc_loss": to_scalar(cur_d_log.get("train/disc_loss")),
                        "train_logits_real": to_scalar(cur_d_log.get("train/logits_real")),
                        "train_logits_fake": to_scalar(cur_d_log.get("train/logits_fake")),
                        "train_latents_std": to_scalar(latents_std),
                    }

                if need_csv_log:
                    csv_logger.log(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "global_step": current_step,
                            "phase": "train_gen" if step_gen else "train_disc",
                            "is_ema": 0,
                            **train_metrics,
                        }
                    )

                if need_live_plot:
                    live_plotter.update(current_step, train_metrics)
                    try:
                        live_plotter.render()
                    except Exception as exc:
                        live_plot_failed = True
                        logger.warning(
                            f"Live loss plot update failed at step {current_step}: {exc}. "
                            "Disable live plotting for the remaining run."
                        )

                if need_wandb_log:
                    wandb_log = {}
                    if cur_g_loss is not None:
                        wandb_log["train/generator_loss"] = cur_g_loss.item()
                    if cur_d_loss is not None:
                        wandb_log["train/discriminator_loss"] = cur_d_loss.item()
                    if cur_g_log.get("train/rec_loss") is not None:
                        wandb_log["train/rec_loss"] = cur_g_log["train/rec_loss"]
                    wandb_log["train/latents_std"] = (
                        latents_std if latents_std is not None else 0.0
                    )
                    wandb.log(wandb_log, step=current_step)

                if current_step % args.eval_steps == 0 or current_step == 1:
                    if global_rank == 0:
                        logger.info("Running validation...")
                    log_validation(model, epoch, batch_idx)
                    if ema is not None:
                        ema.apply_shadow()
                        log_validation(model, epoch, batch_idx, "ema")
                        ema.restore()
                    set_train(modules_to_train)

                if current_step % args.save_ckpt_step == 0 and global_rank == 0:
                    file_path = save_checkpoint(
                        epoch,
                        current_step,
                        {
                            "gen_optimizer": gen_optimizer.state_dict(),
                            "disc_optimizer": disc_optimizer.state_dict(),
                        },
                        {
                            "gen_model": model.module.state_dict(),
                            "disc_model": disc.module.state_dict(),
                        },
                        scaler.state_dict(),
                        ddp_sampler.state_dict(),
                        ckpt_dir,
                        f"checkpoint-{current_step}.ckpt",
                        ema_state_dict=ema.shadow if ema is not None else {},
                    )
                    logger.info(f"Checkpoint saved to `{file_path}`")

                if current_step >= max_steps:
                    break
    finally:
        if csv_logger is not None:
            csv_logger.close()

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed image VAE training")

    parser.add_argument("--exp_name", type=str, default="wfvae2-image")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_ckpt_step", type=int, default=500)
    parser.add_argument("--ckpt_dir", type=str, default="./results/")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--csv_log_steps", type=int, default=200)
    parser.add_argument("--csv_log_path", type=str, default="")
    parser.add_argument("--disable_csv_log", action="store_true")
    parser.add_argument("--enable_live_plot", action="store_true", default=True)
    parser.add_argument("--disable_live_plot", action="store_false", dest="enable_live_plot")
    parser.add_argument("--live_plot_every_steps", type=int, default=200)
    parser.add_argument("--live_plot_path", type=str, default="")
    parser.add_argument("--live_plot_history", type=int, default=0)
    parser.add_argument("--live_plot_dpi", type=int, default=150)
    parser.add_argument("--enable_patch_score_vis", action="store_true", default=True)
    parser.add_argument("--disable_patch_score_vis", action="store_false", dest="enable_patch_score_vis")
    parser.add_argument("--patch_score_vis_max_samples", type=int, default=0)
    parser.add_argument("--patch_score_min_cell_px", type=int, default=32)
    parser.add_argument("--patch_score_save_maps", action="store_true", default=True)
    parser.add_argument("--no_patch_score_save_maps", action="store_false", dest="patch_score_save_maps")
    parser.add_argument("--patch_score_save_summary", action="store_true", default=True)
    parser.add_argument("--no_patch_score_save_summary", action="store_false", dest="patch_score_save_summary")
    parser.add_argument("--patch_score_log_wandb", action="store_true", default=True)
    parser.add_argument("--no_patch_score_log_wandb", action="store_false", dest="patch_score_log_wandb")
    parser.add_argument("--enable_val_image_dump", action="store_true", default=True)
    parser.add_argument("--disable_val_image_dump", action="store_false", dest="enable_val_image_dump")
    parser.add_argument("--val_image_dump_max_samples", type=int, default=0)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--clip_grad_norm", type=float, default=1e5)

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--use_manifest", action="store_true")
    parser.add_argument("--resolution", type=int, default=1024)

    parser.add_argument("--ignore_mismatched_sizes", action="store_true")
    parser.add_argument("--find_unused_parameters", action="store_true")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="WFVAE2Image")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--model_config", type=str, default="examples/wfvae2-image-1024.json")
    parser.add_argument(
        "--mix_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument("--wavelet_loss", action="store_true")
    parser.add_argument("--not_resume_discriminator", action="store_true")
    parser.add_argument("--not_resume_optimizer", action="store_true")
    parser.add_argument("--wavelet_weight", type=float, default=0.1)

    parser.add_argument(
        "--disc_cls",
        type=str,
        default="wfimagevae.model.losses.LPIPSWithDiscriminator2D",
    )
    parser.add_argument("--disc_start", type=int, default=5)
    parser.add_argument("--disc_weight", type=float, default=0.5)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--perceptual_weight", type=float, default=1.0)
    parser.add_argument("--loss_type", type=str, default="l1")
    parser.add_argument("--logvar_init", type=float, default=0.0)

    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--eval_image_path", type=str, required=True)
    parser.add_argument("--eval_resolution", type=int, default=1024)
    parser.add_argument("--eval_crop_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--eval_subset_size", type=int, default=64)
    parser.add_argument("--eval_num_image_log", type=int, default=8)

    parser.add_argument("--dataset_num_worker", type=int, default=8)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    parser.add_argument("--disable_persistent_workers", action="store_true")

    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    args = parser.parse_args()

    set_random_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
