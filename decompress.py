"""python decompress.py compressed_dir output_dir   ->  rebuilds <name>.png for every <name>.bin"""
import sys
from pathlib import Path

from PIL import Image

from imgcomp.runtime import Runtime, unpack

WEIGHTS = Path(__file__).resolve().parent / "weights" / "codec.pt"


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    rt = Runtime(WEIGHTS)
    for p in sorted(src.glob("*.bin")):
        level, h, w, words = unpack(p.read_bytes())
        img = rt.synthesize(rt.decode_latent(words, h, w), level, h, w)
        Image.fromarray(img).save(dst / (p.stem + ".png"))


if __name__ == "__main__":
    main()
