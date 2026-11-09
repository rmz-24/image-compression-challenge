"""python tools/export.py runs/main/last.pt weights/codec.pt  -> deploy weights for compress/decompress"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imgcomp.model import Codec
from imgcomp.runtime import export

ck = torch.load(sys.argv[1], map_location="cpu")
m = Codec()
m.load_state_dict(ck["model"])
Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
export(m, sys.argv[2])
print("exported step", ck["step"], "->", sys.argv[2], f"{Path(sys.argv[2]).stat().st_size / 1e6:.1f} MB")
