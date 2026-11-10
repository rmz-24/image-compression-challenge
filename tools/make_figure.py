"""Side-by-side figure: original vs this codec vs JPEG (same size) vs JPEG (passing 0.95), with zoomed crops.

python tools/make_figure.py kodak_dir compressed_dir reconstructed_dir out.png kodim23:300,40 kodim13:...
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import ms_ssim

CROP, ZOOM, W = 160, 2, 384  # crop size (px), zoom factor, panel width


def jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=q, optimize=True)
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")), len(buf.getvalue())


def jpeg_at_size(img, nbytes):
    best = jpeg(img, 1)
    for q in range(1, 96):
        r = jpeg(img, q)
        if r[1] > nbytes:
            break
        best = r
    return best


def jpeg_passing(img):
    for q in range(1, 96):
        r = jpeg(img, q)
        if ms_ssim(img, r[0]) >= 0.95:
            return r
    return r


def main():
    kodak, comp, rec, out = map(Path, sys.argv[1:5])
    font = ImageFont.truetype("arial.ttf", 17)
    bold = ImageFont.truetype("arialbd.ttf", 19)
    rows = []
    for spec in sys.argv[5:]:
        name, xy = spec.split(":")
        cx, cy = map(int, xy.split(","))
        orig = np.array(Image.open(kodak / f"{name}.png").convert("RGB"))
        h, w = orig.shape[:2]
        ours = np.array(Image.open(rec / f"{name}.png").convert("RGB"))
        nb = (comp / f"{name}.bin").stat().st_size
        j_same, nj = jpeg_at_size(orig, nb)
        j_pass, np_ = jpeg_passing(orig)
        bpp = lambda n: n * 8 / (h * w)
        panels = [
            ("Original", orig, "24 bpp (uncompressed)", (60, 60, 60)),
            ("This codec", ours, f"{bpp(nb):.3f} bpp  ·  MS-SSIM {ms_ssim(orig, ours):.4f}", (20, 120, 50)),
            ("JPEG, same file size", j_same, f"{bpp(nj):.3f} bpp  ·  MS-SSIM {ms_ssim(orig, j_same):.4f}", (170, 40, 40)),
            ("JPEG passing 0.95", j_pass, f"{bpp(np_):.3f} bpp  ·  MS-SSIM {ms_ssim(orig, j_pass):.4f}", (60, 60, 60)),
        ]
        ph = round(W * h / w)
        row = Image.new("RGB", (W * 4 + 50, 40 + ph + CROP * ZOOM * W // (CROP * ZOOM) + 70), "white")
        d = ImageDraw.Draw(row)
        for i, (title, a, cap, col) in enumerate(panels):
            x0 = 10 + i * (W + 10)
            d.text((x0 + W // 2, 22), title, fill=col, font=bold, anchor="mm")
            full = Image.fromarray(a).resize((W, ph), Image.LANCZOS)
            row.paste(full, (x0, 40))
            s = W / w
            ImageDraw.Draw(row).rectangle([x0 + cx * s, 40 + cy * s, x0 + (cx + CROP) * s, 40 + (cy + CROP) * s],
                                          outline=(255, 200, 0), width=2)
            crop = Image.fromarray(a[cy:cy + CROP, cx:cx + CROP]).resize((W, W), Image.NEAREST)
            row.paste(crop, (x0, 44 + ph))
            d.text((x0 + W // 2, 44 + ph + W + 18), cap, fill=col, font=font, anchor="mm")
        rows.append(row)
    fig = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        fig.paste(r, (0, y))
        y += r.height
    fig.save(out)
    print("saved", out, fig.size)


if __name__ == "__main__":
    main()
