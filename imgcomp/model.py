"""Learned image codec: ELIC-style transforms, hyperprior, channel-wise context, gain-based variable rate.

Every network on the entropy-parameter path (h_s and the slice parameter nets) is built from
QConv layers. When quantization is enabled, those layers simulate the fixed-point arithmetic used
at deploy time, where they run in exact integer math so bitstreams decode identically on any machine.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N = 192          # transform width
M = 320          # latent channels
NZ = 192         # hyper-latent channels
SLICES = [16, 16, 32, 64, 192]
LEVELS = 8
LAMBDAS = [1.5 * (64 / 1.5) ** (i / (LEVELS - 1)) for i in range(LEVELS)]

# fixed-point format of the integer entropy-parameter path
FA = 8           # fractional bits of activations
WBITS = 12       # signed weight magnitude bits
ACT_CLAMP = 2.0 ** 14  # activation magnitude clamp (real units)
LEAKY = 0.125

# Gaussian scale table (log domain, in 1/256 units) used at deploy time
LOG_SIG_MIN = math.log(0.11)
SCALE_T0 = round(LOG_SIG_MIN * 256)
SCALE_D = 26
SCALE_K = 64
LOG_SIG_MAX = (SCALE_T0 + SCALE_D * (SCALE_K - 1)) / 256


def ste_round(x):
    return x + (torch.round(x) - x).detach()


class _LowerBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b):
        ctx.save_for_backward(x)
        ctx.b = b
        return x.clamp_min(b)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        pass_through = (x >= ctx.b) | (g < 0)
        return g * pass_through, None


def lower_bound(x, b):
    return _LowerBound.apply(x, b)


def upper_bound(x, b):
    return -_LowerBound.apply(-x, -b)


# ------------------------------------------------------------------ transforms ----

def conv(ci, co, k=5, s=2):
    return nn.Conv2d(ci, co, k, s, k // 2)


def deconv(ci, co, k=5, s=2):
    return nn.ConvTranspose2d(ci, co, k, s, k // 2, output_padding=s - 1)


class ResBottleneck(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c // 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, c // 2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, c, 1))

    def forward(self, x):
        return x + self.body(x)


def _stage(c, n=3):
    return [ResBottleneck(c) for _ in range(n)]


class Analysis(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv(3, N), *_stage(N),
            conv(N, N), *_stage(N),
            conv(N, N), *_stage(N),
            conv(N, M))

    def forward(self, x):
        return self.net(x)


class Synthesis(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            deconv(M, N), *_stage(N),
            deconv(N, N), *_stage(N),
            deconv(N, N), *_stage(N),
            deconv(N, 3))

    def forward(self, y):
        return self.net(y)


class HyperAnalysis(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(M, N, 3, 1, 1), nn.LeakyReLU(inplace=True),
            conv(N, N), nn.LeakyReLU(inplace=True),
            conv(N, NZ))

    def forward(self, y):
        return self.net(y)


# ---------------------------------------------------- integer-capable networks ----

class QConv(nn.Conv2d):
    """Conv2d that can simulate the deploy-time fixed-point arithmetic (fake quantization)."""
    quant = False

    def weight_frac_bits(self):
        m = self.weight.detach().abs().amax(dim=(1, 2, 3)).clamp_min(1e-8)
        return torch.floor(torch.log2((2 ** (WBITS - 1) - 1) / m)).clamp(0, 24)

    def forward(self, x):
        if not self.quant:
            return super().forward(x)
        s = 2.0 ** self.weight_frac_bits()
        w = ste_round(self.weight * s.view(-1, 1, 1, 1)) / s.view(-1, 1, 1, 1)
        bs = s * 2 ** FA
        b = ste_round(self.bias * bs) / bs
        y = F.conv2d(x, w, b, self.stride, self.padding)
        return ste_round(y * 2 ** FA) / 2 ** FA


class QAct(nn.Module):
    def forward(self, x):
        return F.leaky_relu(x, LEAKY).clamp(-ACT_CLAMP, ACT_CLAMP)


class QShuffle(nn.Module):
    def forward(self, x):
        return F.pixel_shuffle(x, 2)


def qconv(ci, co, k=3):
    return QConv(ci, co, k, 1, k // 2)


class HyperSynthesis(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            qconv(NZ, N), QAct(),
            qconv(N, 4 * N), QShuffle(), QAct(),
            qconv(N, 4 * N), QShuffle(), QAct(),
            qconv(N, 2 * M))

    def forward(self, z):
        return self.net(z)


class SliceParams(nn.Module):
    def __init__(self, c_prev, c_out):
        super().__init__()
        self.net = nn.Sequential(
            qconv(2 * M + c_prev, 224), QAct(),
            qconv(224, 128), QAct(),
            qconv(128, 2 * c_out))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------ entropy models ----

class EntropyBottleneck(nn.Module):
    """Ballé factorized density for the hyper-latent (per channel)."""

    def __init__(self, c, filters=(3, 3, 3), init_scale=10.0):
        super().__init__()
        f = (1,) + filters + (1,)
        scale = init_scale ** (1 / (len(f) - 1))
        self.mats = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()
        for i in range(len(f) - 1):
            init = math.log(math.expm1(1 / scale / f[i + 1]))
            self.mats.append(nn.Parameter(torch.full((c, f[i + 1], f[i]), init)))
            self.biases.append(nn.Parameter(torch.rand(c, f[i + 1], 1) - 0.5))
            if i < len(f) - 2:
                self.factors.append(nn.Parameter(torch.zeros(c, f[i + 1], 1)))

    def logits_cdf(self, v):  # v: (C, 1, K)
        for i in range(len(self.mats)):
            v = torch.matmul(F.softplus(self.mats[i]), v) + self.biases[i]
            if i < len(self.factors):
                v = v + torch.tanh(self.factors[i]) * torch.tanh(v)
        return v

    def likelihood(self, z):
        B, C, H, W = z.shape
        v = z.permute(1, 0, 2, 3).reshape(C, 1, -1)
        lo = self.logits_cdf(v - 0.5)
        up = self.logits_cdf(v + 0.5)
        sign = -torch.sign(lo + up).detach()
        lik = (torch.sigmoid(sign * up) - torch.sigmoid(sign * lo)).abs()
        return lik.reshape(C, B, H, W).permute(1, 0, 2, 3)


def gaussian_likelihood(y, mu, log_sig):
    sig = torch.exp(log_sig)
    v = (y - mu).abs()
    c = 1 / (sig * math.sqrt(2))
    up = 0.5 * torch.erfc((v - 0.5) * c)
    lo = 0.5 * torch.erfc((v + 0.5) * c)
    return up - lo


def bound_log_sig(ls):
    return upper_bound(lower_bound(ls, LOG_SIG_MIN), LOG_SIG_MAX)


# ------------------------------------------------------------------- codec ----

class Codec(nn.Module):
    amp = False  # bf16 autocast for the big transforms (training speed)

    def __init__(self):
        super().__init__()
        self.g_a = Analysis()
        self.g_s = Synthesis()
        self.h_a = HyperAnalysis()
        self.h_s = HyperSynthesis()
        self.eb = EntropyBottleneck(NZ)
        cp = [sum(SLICES[:i]) for i in range(len(SLICES))]
        self.params = nn.ModuleList(SliceParams(cp[i], SLICES[i]) for i in range(len(SLICES)))
        lam = torch.tensor(LAMBDAS)
        init = 0.3 * torch.log(lam / lam[LEVELS // 2])
        self.log_gain = nn.Parameter(init[:, None].repeat(1, M))
        self.log_igain = nn.Parameter(-init[:, None].repeat(1, M))

    def set_quant(self, on):
        for m in self.modules():
            if isinstance(m, QConv):
                m.quant = on

    def gains(self, level):
        """level: (B,) float in [0, LEVELS-1]; geometric interpolation between trained gains."""
        lo = level.floor().long().clamp(0, LEVELS - 2)
        t = (level - lo.float())[:, None]
        g = (1 - t) * self.log_gain[lo] + t * self.log_gain[lo + 1]
        gi = (1 - t) * self.log_igain[lo] + t * self.log_igain[lo + 1]
        return torch.exp(g)[:, :, None, None], torch.exp(gi)[:, :, None, None]

    def entropy_params(self, hf, prev, i):
        out = self.params[i](torch.cat([hf] + prev, dim=1))
        mu, ls = out.chunk(2, dim=1)
        return mu, ls

    def forward(self, x, level, noise=True):
        g, gi = self.gains(level)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            ys = self.g_a(x).float() * g
            z = self.h_a(ys).float()
        zn = z + torch.empty_like(z).uniform_(-0.5, 0.5) if noise else torch.round(z)
        lik_z = self.eb.likelihood(zn)
        hf = self.h_s(ste_round(z))
        prev, liks = [], []
        for i, ysl in enumerate(ys.split(SLICES, dim=1)):
            mu, ls = self.entropy_params(hf, prev, i)
            ls = bound_log_sig(ls)
            yq = ste_round(ysl - mu) + mu
            noisy = ysl + torch.empty_like(ysl).uniform_(-0.5, 0.5) if noise else yq
            liks.append(gaussian_likelihood(noisy, mu, ls))
            prev.append(yq)
        y_hat = torch.cat(prev, dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            x_hat = self.g_s(y_hat * gi).float()
        lik_y = torch.cat(liks, dim=1)
        bits = (-torch.log2(lik_y.clamp_min(1e-9))).sum(dim=(1, 2, 3)) + \
               (-torch.log2(lik_z.clamp_min(1e-9))).sum(dim=(1, 2, 3))
        return x_hat, bits
