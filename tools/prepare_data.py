"""Turn DIV2K_train_HR PNGs into uint8 .npy arrays at several scales (multi-scale training data)."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

SCALES = [1.0, 0.75, 0.5, 0.35]


def work(args):
    src, dst = args
    im = Image.open(src).convert("RGB")
    for s in SCALES:
        out = dst / f"{src.stem}_{int(s * 100):03d}.npy"
        if out.exists():
            continue
        r = im if s == 1.0 else im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        np.save(out, np.asarray(r))


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.png"))
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(work, [(f, dst) for f in files]))
    print(len(files), "images done")
