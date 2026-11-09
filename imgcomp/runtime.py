"""Deploy-time codec: weight export, exact integer entropy-parameter nets, range coding.

Bitstream (one file per image):
    byte 0      : rate level index into the fine gain table (0..255)
    bytes 1..4  : height, width (uint16 little endian)
    rest        : range-coder words (uint32 LE), trailing zero bytes stripped

Decoding must reproduce the encoder's probability tables bit-exactly. Everything that feeds the
tables (hyper-synthesis + slice parameter nets) therefore runs in integer arithmetic, carried in
float64 tensors on CPU (all values are integers < 2^53, so every summation order is exact).
"""
import math
import struct

import constriction
import numpy as np
import torch
import torch.nn.functional as F

from .model import (ACT_CLAMP, FA, LEAKY, LEVELS, M, NZ, SCALE_D, SCALE_K, SCALE_T0, SLICES,
                    Analysis, Codec, EntropyBottleneck, HyperAnalysis, HyperSynthesis, QAct,
                    QConv, QShuffle, SliceParams, Synthesis, bound_log_sig, gaussian_likelihood,
                    ste_round)
from .msssim import ms_ssim

FINE_LEVELS = 256
PAD = 64
_CM = constriction.stream.model
_Q = constriction.stream.queue
_ESC_BITS = _CM.Categorical(np.full(32, 1 / 32), perfect=False)  # escape: bit length (0..31)
_BIT = _CM.Categorical(np.array([0.5, 0.5]), perfect=False)


# ------------------------------------------------------------------ export ----

def _export_intnet(seq):
    layers = []
    for m in seq:
        if isinstance(m, QConv):
            fw = m.weight_frac_bits()
            s = 2.0 ** fw
            w = torch.round(m.weight.detach() * s.view(-1, 1, 1, 1))
            assert w.abs().max() < 2 ** 15
            b = torch.round(m.bias.detach().double() * s.double() * 2 ** FA)
            layers.append({"w": w.to(torch.int16).cpu(), "b": b.to(torch.int64).cpu(),
                           "fw": fw.to(torch.int8).cpu(), "pad": m.padding[0]})
        elif isinstance(m, QAct):
            layers.append({"act": True})
        elif isinstance(m, QShuffle):
            layers.append({"shuffle": True})
        else:
            raise TypeError(m)
    return layers


def _norm_cdf(x):
    return 0.5 * np.vectorize(math.erfc)(-np.asarray(x, dtype=np.float64) / math.sqrt(2))


def _gauss_tables():
    tabs = []
    for k in range(SCALE_K):
        sig = math.exp((SCALE_T0 + k * SCALE_D) / 256)
        r = max(2, math.ceil(7 * sig))
        s = np.arange(-r, r + 1)
        p = _norm_cdf((s + 0.5) / sig) - _norm_cdf((s - 0.5) / sig)
        esc = 2 * _norm_cdf(-(r + 0.5) / sig)
        tabs.append(np.append(p, max(esc, 1e-9)))
    return tabs


@torch.no_grad()
def _z_tables(eb):
    """Per-channel pmf of the rounded hyper-latent: (offset, pmf with escape as last entry)."""
    dev = eb.mats[0].device
    grid = torch.arange(-256, 257, device=dev, dtype=torch.float32)
    v = grid.view(1, 1, -1).repeat(NZ, 1, 1)
    lo = torch.sigmoid(eb.logits_cdf(v - 0.5))[:, 0].double().cpu().numpy()
    up = torch.sigmoid(eb.logits_cdf(v + 0.5))[:, 0].double().cpu().numpy()
    tabs = []
    for c in range(NZ):
        p = up[c] - lo[c]
        nz = np.nonzero(p > 1e-8)[0]
        a, b = nz.min(), nz.max()
        pmf = np.maximum(p[a:b + 1], 1e-9)
        esc = max(1 - pmf.sum(), 1e-9)
        tabs.append((int(grid[a]), np.append(pmf, esc)))
    return tabs


@torch.no_grad()
def export(model: Codec, path):
    model = model.float().eval()
    lv = torch.arange(FINE_LEVELS, dtype=torch.float32, device=model.log_gain.device) \
        * (LEVELS - 1) / (FINE_LEVELS - 1)
    g, gi = model.gains(lv)
    half = lambda mod: {k: v.half().cpu() for k, v in mod.state_dict().items()}
    blob = {
        "g_a": half(model.g_a), "h_a": half(model.h_a), "g_s": half(model.g_s),
        "h_s": _export_intnet(model.h_s.net),
        "params": [_export_intnet(p.net) for p in model.params],
        "spatial": [{"ctx": _export_intnet(sp.ctx), "agg": _export_intnet(sp.agg)} for sp in model.spatial],
        "gain": g.view(FINE_LEVELS, M).cpu(), "igain": gi.view(FINE_LEVELS, M).cpu(),
        "ztab": _z_tables(model.eb),
        "eb": {k: v.cpu() for k, v in model.eb.state_dict().items()},
        "ytab": _gauss_tables(),
    }
    torch.save(blob, path)


# ------------------------------------------------------------ integer nets ----

def run_intnet(layers, x):
    """x: float64 CPU tensor of integers (fixed point, FA fractional bits)."""
    lim = ACT_CLAMP * 2 ** FA
    for L in layers:
        if "w" in L:
            acc = F.conv2d(x, L["w64"], L["b64"], padding=L["pad"])
            d = L["div"]
            x = torch.floor((acc + d / 2) / d)
        elif "act" in L:
            x = torch.where(x >= 0, x, torch.floor(x * LEAKY)).clamp(-lim, lim)
        else:
            x = F.pixel_shuffle(x, 2)
    return x


def _prep_intnet(layers):
    for L in layers:
        if "w" in L:
            L["w64"] = L["w"].double()
            L["b64"] = L["b"].double()
            L["div"] = (2.0 ** L["fw"].double()).view(1, -1, 1, 1)
    return layers


# ---------------------------------------------------------------- runtime ----

class Runtime:
    def __init__(self, path, device=None):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.g_a, self.h_a, self.g_s = Analysis(), HyperAnalysis(), Synthesis()
        for name in ("g_a", "h_a", "g_s"):
            mod = getattr(self, name)
            mod.load_state_dict({k: v.float() for k, v in blob[name].items()})
            mod.to(self.dev).eval()
        self.h_s = _prep_intnet(blob["h_s"])
        self.params = [_prep_intnet(p) for p in blob["params"]]
        self.gain, self.igain = blob["gain"], blob["igain"]
        self.zmodels = [(off, _CM.Categorical(p, perfect=False), len(p) - 1) for off, p in blob["ztab"]]
        self.ymodels = [(_CM.Categorical(p, perfect=False), (len(p) - 2) // 2) for p in blob["ytab"]]
        self.spatial = [{k: _prep_intnet(v) for k, v in sp.items()} for sp in blob["spatial"]]
        self.blob = blob
        self._twin = None

    # ---- float twin of the deployed model, for encoder-side latent refinement ----
    def _build_twin(self):
        twin = Codec()
        twin.h_a.load_state_dict(self.h_a.state_dict())
        twin.g_s.load_state_dict(self.g_s.state_dict())
        twin.eb.load_state_dict(self.blob["eb"])
        nets = [(twin.h_s.net, self.blob["h_s"])]
        for i in range(len(SLICES)):
            nets.append((twin.params[i].net, self.blob["params"][i]))
            nets.append((twin.spatial[i].ctx, self.blob["spatial"][i]["ctx"]))
            nets.append((twin.spatial[i].agg, self.blob["spatial"][i]["agg"]))
        for seq, layers in nets:
            convs = [m for m in seq if isinstance(m, QConv)]
            for m, L in zip(convs, [L for L in layers if "w" in L], strict=True):
                s = 2.0 ** L["fw"].float()
                m.weight.data = L["w"].float() / s.view(-1, 1, 1, 1)
                m.bias.data = (L["b"].double() / (s.double() * 2 ** FA)).float()
        twin.to(self.dev).eval()
        for p in twin.parameters():
            p.requires_grad_(False)
        self._twin = twin

    def refine(self, x, y0, level, lam, iters=60, lr=0.002, target=None):
        """Gradient-descend the latent y at a fixed level on bits/px + lam * distortion term.
        target=None: distortion = 1 - MS-SSIM; else relu(target - MS-SSIM)."""
        if self._twin is None:
            self._build_twin()
        x = x.to(self.dev)
        h, w = x.shape[2:]
        g = self.gain[level].to(self.dev).view(1, -1, 1, 1)
        gi = self.igain[level].to(self.dev).view(1, -1, 1, 1)
        y = y0.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([y], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for _ in range(iters):
            x_hat, bits = self._twin.from_latent(y * g, gi, noise=False)
            ms = ms_ssim(x, x_hat.clamp(0, 1))[0]
            dist = 1 - ms if target is None else torch.relu(target - ms)
            loss = bits[0] / (h * w) + lam * dist
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        return y.detach()

    # ---- shared entropy-parameter pass (identical on both sides) ----
    @staticmethod
    def _split(out, c):
        mu, t = out[:, :c], out[:, c:]
        idx = torch.div(t.long() - SCALE_T0 + SCALE_D // 2, SCALE_D, rounding_mode="floor")
        return mu, idx.clamp(0, SCALE_K - 1)

    def _slices(self, hf, step):
        """Entropy-parameter pass shared by encoder and decoder. For each slice, anchors (checkerboard)
        are coded first, then non-anchors with params refined from the decoded anchors.
        step(i, mu, idx, sel) codes the positions sel (bool HxW) and returns symbols (tensor like mu)."""
        h, w = hf.shape[2:]
        sel = ((np.arange(h)[:, None] + np.arange(w)[None, :]) % 2 == 0)
        m = torch.from_numpy(sel).double()[None, None]
        prev = []
        for i, c in enumerate(SLICES):
            out = run_intnet(self.params[i], torch.cat([hf] + prev, dim=1))
            mu, idx = self._split(out, c)
            ya = (mu + step(i, mu, idx, sel) * 2 ** FA) * m
            sp = self.spatial[i]
            out = out + run_intnet(sp["agg"], torch.cat([out, run_intnet(sp["ctx"], ya)], dim=1))
            mu2, idx2 = self._split(out, c)
            yn = mu2 + step(i, mu2, idx2, ~sel) * 2 ** FA
            prev.append(torch.where(m.bool(), ya, yn))
        return torch.cat(prev, dim=1)

    # ---- symbol coding helpers ----
    @staticmethod
    def _enc_escapes(enc, vals):
        for v in vals:
            v = int(v)
            enc.encode(np.array([v < 0], dtype=np.int32), _BIT)
            m = abs(v)
            n = m.bit_length()
            enc.encode(np.array([n], dtype=np.int32), _ESC_BITS)
            if n > 1:
                bits = [(m >> j) & 1 for j in range(n - 1)]
                enc.encode(np.array(bits, dtype=np.int32), _BIT)

    @staticmethod
    def _dec_escapes(dec, count):
        out = []
        for _ in range(count):
            neg = dec.decode(_BIT, 1)[0]
            n = int(dec.decode(_ESC_BITS, 1)[0])
            m = 0
            if n > 1:
                bits = dec.decode(_BIT, n - 1)
                m = sum(int(b) << j for j, b in enumerate(bits))
            m = (m | (1 << (n - 1))) if n > 0 else 0
            out.append(-m if neg else m)
        return np.array(out, dtype=np.int64)

    def _enc_group(self, enc, sym, model, r):
        """sym: int64 values in this group; r: table half range; escape index = 2r+1."""
        u = sym + r
        esc = (u < 0) | (u > 2 * r)
        u = np.where(esc, 2 * r + 1, u)
        enc.encode(u.astype(np.int32), model)
        if esc.any():
            self._enc_escapes(enc, sym[esc])

    def _dec_group(self, dec, n, model, r):
        u = dec.decode(model, n).astype(np.int64)
        sym = u - r
        esc = u == 2 * r + 1
        if esc.any():
            sym[esc] = self._dec_escapes(dec, int(esc.sum()))
        return sym

    def _enc_z(self, enc, zq):
        for c, (off, model, r_esc) in enumerate(self.zmodels):
            s = zq[c].reshape(-1)
            u = s - off
            esc = (u < 0) | (u >= r_esc)
            enc.encode(np.where(esc, r_esc, u).astype(np.int32), model)
            if esc.any():
                self._enc_escapes(enc, s[esc])

    def _dec_z(self, dec, h, w):
        z = np.zeros((NZ, h * w), dtype=np.int64)
        for c, (off, model, r_esc) in enumerate(self.zmodels):
            u = dec.decode(model, h * w).astype(np.int64)
            s = u + off
            esc = u == r_esc
            if esc.any():
                s[esc] = self._dec_escapes(dec, int(esc.sum()))
            z[c] = s
        return z.reshape(NZ, h, w)

    # ---- encoder ----
    @torch.no_grad()
    def analysis(self, x):
        """x: (1,3,H,W) float in [0,1], padded. Returns y (float, on device)."""
        return self.g_a(x.to(self.dev))

    @torch.no_grad()
    def encode_latent(self, y, level):
        """Quantize + entropy-code y at fine level. Returns (range-coder words, y_hat fixed point)."""
        ys = y * self.gain[level].to(y.device).view(1, -1, 1, 1)
        z = self.h_a(ys)
        zq = torch.round(z).double().cpu()
        hf = run_intnet(self.h_s, zq * 2 ** FA)
        ys = ys.double().cpu()
        bounds = np.cumsum([0] + SLICES)
        enc = _Q.RangeEncoder()
        self._enc_z(enc, zq[0].long().numpy())

        def step(i, mu, idx, sel):
            sym = torch.round(ys[:, bounds[i]:bounds[i + 1]] - mu / 2 ** FA)
            sv, iv = sym[0].long().numpy()[:, sel], idx[0].numpy()[:, sel]
            for k in np.unique(iv):
                model, r = self.ymodels[k]
                self._enc_group(enc, sv[iv == k], model, r)
            return sym

        y_hat = self._slices(hf, step)
        return enc.get_compressed(), y_hat

    @torch.no_grad()
    def synthesize(self, yq, level, h, w):
        y = (yq / 2 ** FA).float().to(self.dev) * self.igain[level].to(self.dev).view(1, -1, 1, 1)
        x = self.g_s(y)[:, :, :h, :w].clamp(0, 1)
        return (x * 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()

    # ---- decoder ----
    @torch.no_grad()
    def decode_latent(self, words, H, W):
        hy, wy = -(-H // PAD) * PAD // 16, -(-W // PAD) * PAD // 16
        dec = _Q.RangeDecoder(words)
        zq = torch.from_numpy(self._dec_z(dec, hy // 4, wy // 4)).double()[None]
        hf = run_intnet(self.h_s, zq * 2 ** FA)

        def step(i, mu, idx, sel):
            iv = idx[0].numpy()[:, sel]
            sv = np.zeros(iv.shape, dtype=np.int64)
            for k in np.unique(iv):
                model, r = self.ymodels[k]
                g = iv == k
                sv[g] = self._dec_group(dec, int(g.sum()), model, r)
            sym = np.zeros(idx.shape[1:], dtype=np.int64)
            sym[:, sel] = sv
            return torch.from_numpy(sym).double()[None]

        return self._slices(hf, step)


# ------------------------------------------------------------- bitstream ----

def pack(level, h, w, words):
    data = words.astype("<u4").tobytes().rstrip(b"\0")
    return struct.pack("<BHH", level, h, w) + data


def unpack(buf):
    level, h, w = struct.unpack("<BHH", buf[:5])
    data = buf[5:]
    data += b"\0" * (-len(data) % 4)
    words = np.frombuffer(data, dtype="<u4").astype(np.uint32)
    return level, h, w, words


def pad_image(img):
    """img: HxWx3 uint8 -> (1,3,Hp,Wp) float tensor with replicate padding to a multiple of PAD."""
    h, w = img.shape[:2]
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    ph, pw = -h % PAD, -w % PAD
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x
