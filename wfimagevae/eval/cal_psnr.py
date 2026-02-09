import numpy as np
import torch

def calculate_psnr(video_recon, inputs, device=None):
    mse = torch.mean(torch.square(inputs - video_recon), dim=(1,2,3))
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    psnr = psnr.mean().detach()
    if psnr == torch.inf:
        return 100
    return psnr.cpu().item()
