"""Image Compression Challenge - evaluation script.

Scores a submission on a folder of original PNG images:
  1. Size: total bytes of the compressed files -> bits per pixel (bpp).
  2. Quality: MS-SSIM of each rebuilt image vs its original.
  3. Gate: EVERY image must reach MS-SSIM >= 0.95, otherwise the submission fails.
Ranking: among passing submissions, lowest average bpp wins.

Two ways to use it:

  # A) You already ran compress + decompress yourself
  python evaluate.py --originals kodak/ --compressed out/compressed/ --reconstructed out/rebuilt/

  # B) Let the script run the submission's commands for you
  python evaluate.py --originals kodak/ --workdir out/ \
      --compress-cmd "python compress.py {input} {output}" \
      --decompress-cmd "python decompress.py {input} {output}"

Works with ANY approach: the script only looks at the outputs.
  - Size = every byte in the compressed folder, whatever the layout
    (one file per image, one archive for everything, sub-folders...).
  - Quality = rebuilt PNGs, which must keep the original file names and sizes
    (e.g. kodim01.png -> kodim01.png).

Requirements: numpy, pillow, scipy
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

THRESHOLD = 0.95
IMG_EXTS = {".png"}

# ---------------------------------------------------------------- MS-SSIM ----
# Standard MS-SSIM (Wang et al. 2003): 11x11 Gaussian window, sigma 1.5,
# 5 scales, valid convolution, 2x2 average-pool between scales,
# computed per RGB channel and averaged. Matches common implementations
# such as pytorch_msssim (data_range=255).

_WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
_C1 = (0.01 * 255) ** 2
_C2 = (0.03 * 255) ** 2


def _gauss(size=11, sigma=1.5):
    x = np.arange(size) - size // 2
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


_G = _gauss()


def _filt(a):
    a = convolve2d(a, _G[None, :], mode="valid")
    return convolve2d(a, _G[:, None], mode="valid")


def _ssim_cs(x, y):
    mx, my = _filt(x), _filt(y)
    sxx = _filt(x * x) - mx * mx
    syy = _filt(y * y) - my * my
    sxy = _filt(x * y) - mx * my
    cs = (2 * sxy + _C2) / (sxx + syy + _C2)
    ssim = ((2 * mx * my + _C1) / (mx * mx + my * my + _C1)) * cs
    return ssim.mean(), cs.mean()


def _pool(a):
    if a.shape[0] % 2:
        a = np.pad(a, ((0, 1), (0, 0)), mode="edge")
    if a.shape[1] % 2:
        a = np.pad(a, ((0, 0), (0, 1)), mode="edge")
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3))


def _ms_ssim_channel(x, y):
    vals = []
    for i in range(5):
        ssim, cs = _ssim_cs(x, y)
        vals.append(ssim if i == 4 else cs)
        if i < 4:
            x, y = _pool(x), _pool(y)
    vals = np.maximum(np.array(vals), 0.0)
    return float(np.prod(vals ** _WEIGHTS))


def ms_ssim(a, b):
    """MS-SSIM between two uint8 RGB images of the same shape (0..1, 1 = identical)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if min(a.shape[:2]) < 11 * 2 ** 4:
        raise ValueError("image too small for 5-scale MS-SSIM (needs >= 176 px per side)")
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.mean([_ms_ssim_channel(a[..., c], b[..., c]) for c in range(3)]))


# ------------------------------------------------------------- helpers ----

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def run(cmd_template, inp, out):
    out.mkdir(parents=True, exist_ok=True)
    cmd = cmd_template.format(input=shlex.quote(str(inp)), output=shlex.quote(str(out)))
    t0 = time.time()
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        sys.exit(f"Command failed ({proc.returncode}): {cmd}")
    return time.time() - t0


def evaluate(originals, compressed, reconstructed):
    orig_files = sorted(p for p in originals.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not orig_files:
        sys.exit(f"No PNG images found in {originals}")

    # Size = EVERYTHING in the compressed folder, whatever its layout:
    # one file per image, one big archive, sub-folders... all of it counts.
    comp_files = [p for p in compressed.rglob("*") if p.is_file()]
    if not comp_files:
        sys.exit(f"No compressed files found in {compressed}")
    total_bits = sum(p.stat().st_size for p in comp_files) * 8

    # Per-image size is shown only when a file name matches an image (info only).
    comp_by_stem = {}
    for p in comp_files:
        comp_by_stem.setdefault(p.stem, []).append(p)

    rows, problems = [], []
    total_pixels = sum(Image.open(op).size[0] * Image.open(op).size[1] for op in orig_files)
    for op in orig_files:
        orig = load_rgb(op)
        h, w = orig.shape[:2]
        rp = reconstructed / op.name
        if not rp.exists():
            problems.append(f"{op.name}: no rebuilt image found")
            continue
        rec = load_rgb(rp)
        if rec.shape != orig.shape:
            problems.append(f"{op.name}: rebuilt size {rec.shape} != original {orig.shape}")
            continue
        score = ms_ssim(orig, rec)
        matched = comp_by_stem.get(op.stem)
        nbytes = sum(p.stat().st_size for p in matched) if matched else None
        rows.append({"image": op.name, "bytes": nbytes,
                     "bpp": nbytes * 8 / (h * w) if matched else None,
                     "ms_ssim": score, "pass": score >= THRESHOLD})

    avg_bpp = total_bits / total_pixels
    passed = not problems and all(r["pass"] for r in rows)
    return {
        "images": rows,
        "problems": problems,
        "compressed_files": len(comp_files),
        "total_compressed_bytes": total_bits // 8,
        "avg_bpp": avg_bpp,
        "min_ms_ssim": min((r["ms_ssim"] for r in rows), default=float("nan")),
        "mean_ms_ssim": float(np.mean([r["ms_ssim"] for r in rows])) if rows else float("nan"),
        "threshold": THRESHOLD,
        "passed": passed,
    }


def print_report(res, times=None):
    print(f"\n{'image':<16}{'bytes':>10}{'bpp':>9}{'MS-SSIM':>10}  gate")
    print("-" * 52)
    for r in res["images"]:
        mark = "ok" if r["pass"] else "FAIL"
        b = f"{r['bytes']:>10}" if r["bytes"] is not None else f"{'-':>10}"
        bpp = f"{r['bpp']:>9.3f}" if r["bpp"] is not None else f"{'-':>9}"
        print(f"{r['image']:<16}{b}{bpp}{r['ms_ssim']:>10.4f}  {mark}")
    print("-" * 52)
    for p in res["problems"]:
        print(f"PROBLEM: {p}")
    print(f"Compressed folder: {res['compressed_files']} file(s), "
          f"{res['total_compressed_bytes']} bytes in total (all of it counts)")
    print(f"Average bpp      : {res['avg_bpp']:.4f}   <- ranking score (lower is better)")
    print(f"Mean MS-SSIM     : {res['mean_ms_ssim']:.4f}   <- tie-breaker (higher is better)")
    print(f"Lowest MS-SSIM   : {res['min_ms_ssim']:.4f}   (must be >= {res['threshold']})")
    if times:
        print(f"Compress time    : {times[0]:.1f} s")
        print(f"Decompress time  : {times[1]:.1f} s")
    verdict = "PASSED" if res["passed"] else "FAILED (an image is below the threshold or missing)"
    print(f"Result           : {verdict}\n")


def main():
    ap = argparse.ArgumentParser(description="Evaluate an image compression submission.")
    ap.add_argument("--originals", required=True, type=Path, help="folder of original PNGs")
    ap.add_argument("--compressed", type=Path, help="folder of compressed files")
    ap.add_argument("--reconstructed", type=Path, help="folder of rebuilt PNGs")
    ap.add_argument("--workdir", type=Path, help="where to write outputs when running commands")
    ap.add_argument("--compress-cmd", help='e.g. "python compress.py {input} {output}"')
    ap.add_argument("--decompress-cmd", help='e.g. "python decompress.py {input} {output}"')
    ap.add_argument("--json", type=Path, help="also save results as JSON")
    args = ap.parse_args()

    times = None
    if args.compress_cmd or args.decompress_cmd:
        if not (args.compress_cmd and args.decompress_cmd and args.workdir):
            sys.exit("--compress-cmd, --decompress-cmd and --workdir go together")
        args.compressed = args.workdir / "compressed"
        args.reconstructed = args.workdir / "reconstructed"
        t1 = run(args.compress_cmd, args.originals, args.compressed)
        t2 = run(args.decompress_cmd, args.compressed, args.reconstructed)
        times = (t1, t2)
    elif not (args.compressed and args.reconstructed):
        sys.exit("Give --compressed and --reconstructed, or the two commands plus --workdir")

    res = evaluate(args.originals, args.compressed, args.reconstructed)
    print_report(res, times)
    if args.json:
        args.json.write_text(json.dumps(res, indent=2))
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
