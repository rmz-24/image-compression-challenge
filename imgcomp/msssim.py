"""Differentiable MS-SSIM that matches evaluate.py (per-RGB-channel, valid conv, averaged)."""
import torch
import torch.nn.functional as F

_WEIGHTS = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def _gauss(size=11, sigma=1.5):
    x = torch.arange(size, dtype=torch.float64) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (g / g.sum()).float()


_G = _gauss()


def _filt(a, g):
    a = F.conv2d(a, g.view(1, 1, 1, -1))
    return F.conv2d(a, g.view(1, 1, -1, 1))


def _pool(a):
    if a.shape[-2] % 2 or a.shape[-1] % 2:
        a = F.pad(a, (0, a.shape[-1] % 2, 0, a.shape[-2] % 2), mode="replicate")
    return F.avg_pool2d(a, 2)


def ms_ssim(x, y):
    """x, y: (B, 3, H, W) in [0, 1]. Returns (B,) MS-SSIM averaged over RGB channels."""
    B, C, H, W = x.shape
    g = _G.to(x.device, x.dtype)
    X = x.reshape(B * C, 1, H, W)
    Y = y.reshape(B * C, 1, H, W)
    vals = []
    for i in range(5):
        mx, my = _filt(X, g), _filt(Y, g)
        sxx = _filt(X * X, g) - mx * mx
        syy = _filt(Y * Y, g) - my * my
        sxy = _filt(X * Y, g) - mx * my
        cs = (2 * sxy + _C2) / (sxx + syy + _C2)
        if i < 4:
            vals.append(cs.mean(dim=(1, 2, 3)))
            X, Y = _pool(X), _pool(Y)
        else:
            ssim = ((2 * mx * my + _C1) / (mx * mx + my * my + _C1)) * cs
            vals.append(ssim.mean(dim=(1, 2, 3)))
    v = torch.stack(vals, dim=-1).clamp_min(1e-6)
    ms = torch.prod(v ** _WEIGHTS.to(x.device, x.dtype), dim=-1)
    return ms.view(B, C).mean(dim=1)
