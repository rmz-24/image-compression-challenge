"""Train the codec from scratch on DIV2K (multi-scale crops prepared by tools/prepare_data.py).

Loss per image: bpp + lambda(level) * (1 - MS-SSIM), with the rate level sampled per image so
one model covers the whole rate range.

    python tools/prepare_data.py data/DIV2K_train_HR data/div2k_npy
    python train.py --data data/div2k_npy --kodak data/kodak --out runs/main --steps 400000
    python train.py ... --resume runs/main/last.pt --quant --steps 420000   # integer-aware finetune
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from imgcomp.model import LAMBDAS, LEVELS, Codec
from imgcomp.msssim import ms_ssim


class Crops(Dataset):
    def __init__(self, root, crop):
        self.files = sorted(Path(root).glob("*.npy"))
        self.crop = crop
        self.arrs = None

    def __len__(self):
        return 10 ** 9

    def __getitem__(self, _):
        if self.arrs is None:
            arrs = [np.load(f, mmap_mode="r") for f in self.files]
            self.arrs = [a for a in arrs if min(a.shape[:2]) >= self.crop]
        a = random.choice(self.arrs)
        h, w = a.shape[:2]
        c = self.crop
        y, x = random.randint(0, h - c), random.randint(0, w - c)
        p = np.array(a[y:y + c, x:x + c])
        if random.random() < 0.5:
            p = p[:, ::-1]
        if random.random() < 0.5:
            p = p[::-1]
        return torch.from_numpy(np.ascontiguousarray(p)).permute(2, 0, 1)


def worker_init(i):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)


def level_lambda(level):
    return LAMBDAS[0] * (LAMBDAS[-1] / LAMBDAS[0]) ** (level / (LEVELS - 1))


@torch.no_grad()
def eval_kodak(model, imgs, dev):
    """Estimated bits (from likelihoods, rounded latents) at a grid of levels; picks the
    smallest level per image with MS-SSIM >= 0.95. Returns (gate bpp, per-level table)."""
    model.eval()
    grid = [i * 0.5 for i in range(2 * (LEVELS - 1) + 1)]
    table = []
    for x in imgs:
        x = x.to(dev)[None]
        row = []
        for lv in grid:
            x_hat, bits = model(x, torch.tensor([lv], device=dev), noise=False)
            q = ms_ssim(x, (x_hat.clamp(0, 1) * 255).round() / 255).item()
            row.append((bits.item() / (x.shape[2] * x.shape[3]), q))
        table.append(row)
    gate = []
    for row in table:
        ok = [b for b, q in row if q >= 0.95]
        gate.append(min(ok) if ok else row[-1][0] * 10)
    model.train()
    per_level = [(lv, np.mean([r[i][0] for r in table]), np.mean([r[i][1] for r in table]),
                  min(r[i][1] for r in table)) for i, lv in enumerate(grid)]
    return float(np.mean(gate)), per_level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/div2k_npy")
    ap.add_argument("--kodak", default="data/kodak")
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--decay-at", type=float, nargs="*", default=[0.8, 0.95])
    ap.add_argument("--resume", default=None)
    ap.add_argument("--reset-opt", action="store_true")
    ap.add_argument("--quant", action="store_true", help="simulate integer entropy-parameter nets")
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = Codec().to(dev, memory_format=torch.channels_last)
    model.amp = True
    model.set_quant(args.quant)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print("missing keys:", len(missing), "unexpected:", unexpected)
        if not args.reset_opt and "opt" in ck:
            opt.load_state_dict(ck["opt"])
        step = ck["step"]
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, start step {step}")

    kodak = [torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1).float() / 255
             for p in sorted(Path(args.kodak).glob("*.png"))]
    dl = DataLoader(Crops(args.data, args.crop), batch_size=args.batch, num_workers=args.workers,
                    worker_init_fn=worker_init, pin_memory=True, persistent_workers=True,
                    prefetch_factor=4)
    log = open(out / "log.txt", "a")

    def lr_at(s):
        f = 1.0
        for d in args.decay_at:
            if s >= d * args.steps:
                f *= 0.1
        return args.lr * f

    t0, acc = time.time(), np.zeros(4)
    model.train()
    for xb in dl:
        if step >= args.steps:
            break
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(step)
        x = (xb.to(dev, non_blocking=True).float() / 255).contiguous(memory_format=torch.channels_last)
        B = x.shape[0]
        level = torch.rand(B, device=dev) * (LEVELS - 1)
        lam = level_lambda(level)
        x_hat, bits = model(x, level)
        x_hat = x_hat.clamp(0, 1)
        bpp = bits / (x.shape[2] * x.shape[3])
        ms = ms_ssim(x, x_hat)
        dist = 1 - ms
        if step < 5000:  # MSE warmup stabilizes early MS-SSIM training
            dist = dist + (1 - step / 5000) * 50 * ((x - x_hat) ** 2).mean(dim=(1, 2, 3))
        loss = (bpp + lam * dist).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        acc += np.array([loss.item(), bpp.mean().item(), ms.mean().item(), 1])

        if step % 200 == 0:
            a = acc[:3] / acc[3]
            msg = (f"step {step} loss {a[0]:.4f} bpp {a[1]:.4f} msssim {a[2]:.4f} "
                   f"lr {lr_at(step):.1e} {(time.time() - t0) / acc[3]:.3f}s/it")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            t0, acc = time.time(), np.zeros(4)
        if step % 2000 == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
                       out / "last.pt")
        if step % 50000 == 0:
            torch.save({"model": model.state_dict(), "step": step}, out / f"step{step}.pt")
        if step % args.eval_every == 0 or step == args.steps:
            gate, per_level = eval_kodak(model, kodak, dev)
            msg = f"EVAL step {step} kodak gate-bpp {gate:.4f} | " + " ".join(
                f"L{lv:.1f}:{b:.3f}/{q:.4f}/{mn:.3f}" for lv, b, q, mn in per_level)
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            t0 = time.time()


if __name__ == "__main__":
    main()
