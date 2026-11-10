"""python compress.py input_dir output_dir

For every PNG, picks the lowest rate level whose decoded image still reaches MS-SSIM >= 0.95
(checked with the exact decode path), then writes <name>.bin.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imgcomp.msssim import ms_ssim
from imgcomp.model import LAMBDAS, LEVELS
from imgcomp.runtime import FINE_LEVELS, Runtime, pack, pad_image

WEIGHTS = Path(__file__).resolve().parent / "weights" / "codec.pt"
TARGET = 0.95
REFINE_LAMBDA = 100.0   # weight of the quality-shortfall term during latent refinement
REFINE_SLACK = 0.0002   # refinement aims slightly above the gate (its rate/quality model is approximate)
REFINE_WINDOW = 48      # refined search covers levels [l0 - window, l0]
MARGIN = 0.00005  # headroom for float differences between machines in the synthesis transform (measured ~1e-7)


def score(rt, ref, out):
    t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].double().to(rt.dev) / 255
    return ms_ssim(t(ref), t(out)).item()


def level_lambda(level):
    t = level * (LEVELS - 1) / (FINE_LEVELS - 1) / (LEVELS - 1)
    return LAMBDAS[0] * (LAMBDAS[-1] / LAMBDAS[0]) ** t


def smallest_passing(trial, lo, hi):
    """Smallest level in [lo, hi] whose trial passes (quality rises with level); hi if none."""
    while lo < hi:
        mid = (lo + hi) // 2
        if trial(mid)[1] >= TARGET + MARGIN:
            hi = mid
        else:
            lo = mid + 1
    return lo


def compress_image(rt, img, refine_iters=0):
    h, w = img.shape[:2]
    x = pad_image(img)
    y = rt.analysis(x)
    passing = []  # (bytes, level, quality, words)

    def make_trial(refine):
        cache = {}

        def trial(level):
            if level not in cache:
                yl = rt.refine(x, y, level, REFINE_LAMBDA, refine_iters,
                               target=TARGET + MARGIN + REFINE_SLACK) if refine else y
                words, yq = rt.encode_latent(yl, level)
                q = score(rt, img, rt.synthesize(yq, level, h, w))
                cache[level] = (words, q)
                if q >= TARGET + MARGIN:
                    passing.append((len(pack(level, h, w, words)), level, q, words))
            return cache[level]
        return trial

    plain = make_trial(False)
    l0 = smallest_passing(plain, 0, FINE_LEVELS - 1)
    for lv in range(max(0, l0 - 6), l0 + 1):  # quality is only roughly monotonic in level
        plain(lv)
    if refine_iters:
        smallest_passing(make_trial(True), max(0, l0 - REFINE_WINDOW), l0)
    if not passing:  # nothing reaches the target: keep the best quality available
        words, q = plain(FINE_LEVELS - 1)
        return pack(FINE_LEVELS - 1, h, w, words), FINE_LEVELS - 1, q
    n, level, q, words = min(passing, key=lambda t: t[0])
    return pack(level, h, w, words), level, q


def main():
    # strip quotes left by shlex.quote when a command line goes through cmd.exe on Windows
    src, dst = (Path(a.strip("'\"")) for a in sys.argv[1:3])
    dst.mkdir(parents=True, exist_ok=True)
    refine_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rt = Runtime(WEIGHTS)
    for p in sorted(src.glob("*.png")):
        img = np.array(Image.open(p).convert("RGB"))
        buf, level, q = compress_image(rt, img, refine_iters)
        (dst / (p.stem + ".bin")).write_bytes(buf)
        print(f"{p.name}: level {level} ms-ssim {q:.4f} {len(buf)} bytes "
              f"{len(buf) * 8 / (img.shape[0] * img.shape[1]):.4f} bpp", flush=True)


if __name__ == "__main__":
    main()
