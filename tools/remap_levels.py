"""Re-target the rate levels of a checkpoint to a new lambda range (one-off, used once during training).

python tools/remap_levels.py in.pt out.pt OLD_MIN OLD_MAX
The new range is imgcomp.model.LAMBDAS. Gain rows are interpolated (linearly extrapolated) in
log-gain space at the old fractional level that had the same lambda.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imgcomp.model import LAMBDAS, LEVELS

ck = torch.load(sys.argv[1], map_location="cpu")
old_min, old_max = float(sys.argv[3]), float(sys.argv[4])
sd = ck["model"]
for key in ("log_gain", "log_igain"):
    old = sd[key]
    rows = []
    for lam in LAMBDAS:
        f = (LEVELS - 1) * math.log(lam / old_min) / math.log(old_max / old_min)
        lo = min(max(int(math.floor(f)), 0), LEVELS - 2)
        t = f - lo
        rows.append((1 - t) * old[lo] + t * old[lo + 1])
    sd[key] = torch.stack(rows)
ck.pop("opt", None)
torch.save(ck, sys.argv[2])
print("remapped", sys.argv[1], "->", sys.argv[2], "lambdas", [round(l, 3) for l in LAMBDAS])
