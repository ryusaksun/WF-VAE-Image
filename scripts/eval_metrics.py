import argparse
import os
from glob import glob

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import sys

sys.path.append(".")
from wfimagevae.eval.cal_lpips import calculate_lpips
from wfimagevae.eval.cal_psnr import calculate_psnr
from wfimagevae.eval.cal_ssim import calculate_ssim


class EvalImagePairDataset(Dataset):
    image_exts = ("jpg", "jpeg", "png", "webp", "bmp")

    def __init__(self, real_image_dir, generated_image_dir, resolution=1024, crop_size=1024):
        self.real_image_dir = real_image_dir
        self.generated_image_dir = generated_image_dir
        self.generated_files = self._scan_images(generated_image_dir, skip_origin=True)
        self.file_pairs = []

        missing_pairs = []
        for generated_file in self.generated_files:
            rel_path = os.path.relpath(generated_file, generated_image_dir)
            real_file = os.path.join(real_image_dir, rel_path)
            if not os.path.exists(real_file):
                missing_pairs.append(rel_path)
                continue
            self.file_pairs.append((real_file, generated_file))

        if missing_pairs:
            preview = ", ".join(missing_pairs[:8])
            raise FileNotFoundError(
                f"Missing {len(missing_pairs)} paired images in real dir. "
                f"Examples: {preview}"
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
            ]
        )

    def _scan_images(self, folder, skip_origin=False):
        files = []
        for ext in self.image_exts:
            files.extend(glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
            files.extend(
                glob(os.path.join(folder, "**", f"*.{ext.upper()}"), recursive=True)
            )
        files = sorted(files)
        if not skip_origin:
            return files
        # Skip only top-level dumped originals under `<generated_image_dir>/origin/`.
        filtered_files = []
        for path in files:
            rel_path = os.path.relpath(path, folder)
            if rel_path == "origin" or rel_path.startswith(f"origin{os.sep}"):
                continue
            filtered_files.append(path)
        return filtered_files

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, index):
        real_path, generated_path = self.file_pairs[index]
        real = Image.open(real_path).convert("RGB")
        generated = Image.open(generated_path).convert("RGB")
        return {
            "real": self.transform(real),
            "generated": self.transform(generated),
        }


def calculate_metric(metric, generated, real, device):
    if metric == "psnr":
        return calculate_psnr(generated, real)
    if metric == "ssim":
        return np.mean(
            list(calculate_ssim(generated, real)["value"].values())
        )
    if metric == "lpips":
        return calculate_lpips(generated, real, device)
    raise NotImplementedError(metric)


def main(args):
    accelerator = Accelerator()

    dataset = EvalImagePairDataset(
        args.real_image_dir,
        args.generated_image_dir,
        resolution=args.resolution,
        crop_size=args.crop_size,
    )
    if args.subset_size:
        dataset = Subset(dataset, indices=list(range(min(args.subset_size, len(dataset)))))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dataloader = accelerator.prepare(dataloader)

    local_weighted_sum = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
    local_sample_count = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        real = batch["real"].to(accelerator.device)
        generated = batch["generated"].to(accelerator.device)
        score = calculate_metric(args.metric, generated, real, accelerator.device)
        batch_size = float(real.shape[0])
        score_tensor = torch.tensor(float(score), device=accelerator.device, dtype=torch.float32)
        local_weighted_sum += score_tensor * batch_size
        local_sample_count += batch_size

    if local_sample_count.item() <= 0:
        if accelerator.is_main_process:
            print("nan")
        return

    local_stats = torch.stack([local_weighted_sum, local_sample_count], dim=0).unsqueeze(0)
    gathered_stats = accelerator.gather_for_metrics(local_stats)

    if accelerator.is_main_process:
        total_weighted_sum = gathered_stats[:, 0].sum()
        total_sample_count = gathered_stats[:, 1].sum()
        if total_sample_count.item() <= 0:
            print("nan")
        else:
            print((total_weighted_sum / total_sample_count).item())


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image reconstruction metrics")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--real_image_dir", type=str, required=True)
    parser.add_argument("--generated_image_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--subset_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--metric",
        type=str,
        default="lpips",
        choices=["psnr", "ssim", "lpips"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
