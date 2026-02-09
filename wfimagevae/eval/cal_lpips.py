import torch
import lpips

spatial = True
loss_fn = lpips.LPIPS(net='alex', spatial=spatial)

def trans(x):
    if x.shape[-3] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = x * 2 - 1
    return x

def calculate_lpips(video_recon, inputs, device):
    loss_fn.to(device)
    video_recon = trans(video_recon)
    inputs = trans(inputs)
    lpips_score = loss_fn.forward(inputs, video_recon).mean().detach().cpu().item()
    return lpips_score
