"""Reference only: smallest JPEG per image that passes MS-SSIM >= 0.95."""
import io, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import ms_ssim

tot_bits = tot_px = 0
for p in sorted(Path(sys.argv[1]).glob("*.png")):
    x = np.array(Image.open(p).convert("RGB"))
    lo, hi = 1, 95
    while lo < hi:
        q = (lo + hi) // 2
        buf = io.BytesIO(); Image.fromarray(x).save(buf, "JPEG", quality=q, optimize=True)
        y = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        if ms_ssim(x, y) >= 0.95: hi = q
        else: lo = q + 1
    buf = io.BytesIO(); Image.fromarray(x).save(buf, "JPEG", quality=lo, optimize=True)
    n = len(buf.getvalue()) * 8
    tot_bits += n; tot_px += x.shape[0] * x.shape[1]
    print(p.name, lo, f"{n / (x.shape[0]*x.shape[1]):.3f}")
print("avg bpp", tot_bits / tot_px)
