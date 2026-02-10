import torch
import lpips

spatial = True
loss_fn = lpips.LPIPS(net='alex', spatial=spatial)
_loss_fn_device = None

def trans(x):
    if x.shape[-3] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = x * 2 - 1
    return x


def _ensure_loss_fn_on_device(device):
    global _loss_fn_device
    target_device = torch.device(device)
    if _loss_fn_device != target_device:
        loss_fn.to(target_device)
        _loss_fn_device = target_device


def calculate_lpips(video_recon, inputs, device):
    _ensure_loss_fn_on_device(device)
    video_recon = trans(video_recon)
    inputs = trans(inputs)
    lpips_score = loss_fn.forward(inputs, video_recon).mean().detach().cpu().item()
    return lpips_score
