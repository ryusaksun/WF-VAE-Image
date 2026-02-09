import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import sys

sys.path.append(".")
from wfimagevae.model import *


def preprocess(image: Image.Image, resolution: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(
                resolution,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )
    return transform(image).unsqueeze(0)


@torch.no_grad()
def main(args: argparse.Namespace):
    data_type = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    if args.mixed_precision == "fp32":
        data_type = torch.float32

    model_cls = ModelRegistry.get_model(args.model_name)
    vae = model_cls.from_pretrained(args.from_pretrained)
    vae = vae.to(args.device, dtype=data_type)
    vae.eval()

    x = preprocess(Image.open(args.image_path).convert("RGB"), args.resolution)
    x = x.to(args.device, dtype=data_type)

    latents = vae.encode(x).latent_dist.sample().to(dtype=data_type)
    image_recon = vae.decode(latents).sample
    image_recon = image_recon[0].float().clamp(-1, 1)

    image_recon = ((image_recon + 1) / 2).cpu().numpy()
    image_recon = np.transpose(image_recon, (1, 2, 0))
    image_recon = (255 * image_recon).astype(np.uint8)
    Image.fromarray(image_recon).save(args.rec_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-image VAE reconstruction")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--rec_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="WFVAE2Image")
    parser.add_argument("--from_pretrained", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
    )
    main(parser.parse_args())
