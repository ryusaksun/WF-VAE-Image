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
from wfimagevae.model import *


class ReconImageDataset(Dataset):
    image_exts = ("jpg", "jpeg", "png", "webp", "bmp")

    def __init__(self, real_image_dir, resolution=1024, crop_size=1024):
        self.real_image_dir = real_image_dir
        self.image_files = self._scan_images(real_image_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _scan_images(self, folder):
        files = []
        for ext in self.image_exts:
            files.extend(glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
            files.extend(
                glob(os.path.join(folder, "**", f"*.{ext.upper()}"), recursive=True)
            )
        return sorted(files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("RGB")
        relative_path = os.path.relpath(image_path, self.real_image_dir)
        return {
            "image": self.transform(image),
            "file_name": relative_path,
        }


def tensor_to_pil(image_tensor):
    image_np = image_tensor.detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (255.0 * image_np).astype(np.uint8)
    return Image.fromarray(image_np)


@torch.no_grad()
def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    data_type = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    if args.mixed_precision == "fp32":
        data_type = torch.float32

    os.makedirs(args.generated_image_dir, exist_ok=True)
    if args.output_origin:
        os.makedirs(os.path.join(args.generated_image_dir, "origin"), exist_ok=True)

    model_cls = ModelRegistry.get_model(args.model_name)
    vae = model_cls.from_pretrained(args.from_pretrained)
    vae = vae.to(device=device, dtype=data_type)
    vae.eval()

    dataset = ReconImageDataset(
        real_image_dir=args.real_image_dir,
        resolution=args.resolution,
        crop_size=args.crop_size,
    )
    if args.subset_size and args.subset_size > 0:
        dataset = Subset(dataset, indices=list(range(min(args.subset_size, len(dataset)))))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    dataloader = accelerator.prepare(dataloader)

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        x = batch["image"].to(device=device, dtype=data_type)

        latents = vae.encode(x).latent_dist.sample().to(data_type)
        image_recon = vae.decode(latents).sample
        image_recon = ((image_recon.float().clamp(-1, 1) + 1) / 2).contiguous()
        image_input = ((x.float().clamp(-1, 1) + 1) / 2).contiguous()

        for idx in range(image_recon.shape[0]):
            relative_path = batch["file_name"][idx]
            output_path = os.path.join(args.generated_image_dir, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tensor_to_pil(image_recon[idx]).save(output_path)

            if args.output_origin:
                origin_path = os.path.join(args.generated_image_dir, "origin", relative_path)
                os.makedirs(os.path.dirname(origin_path), exist_ok=True)
                tensor_to_pil(image_input[idx]).save(origin_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch image reconstruction with VAE")
    parser.add_argument("--real_image_dir", type=str, required=True)
    parser.add_argument("--generated_image_dir", type=str, required=True)
    parser.add_argument("--from_pretrained", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="WFVAE2Image")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--subset_size", type=int, default=0)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument("--output_origin", action="store_true")
    main(parser.parse_args())
