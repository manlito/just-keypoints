"""Time jkp backends and the original model (PyTorch) on one image at several sizes.

    python tools/bench.py --jkp "--backend metal" --jkp "--backend metal --precision fp16" --torch mps
    python tools/bench.py --jkp "--backend cuda" --jkp "--backend cuda --precision fp16" --torch cuda
    python tools/bench.py --model xfeat --jkp "--backend metal" --torch mps     # XFeat (detectAndCompute)

Each column is timed in turn, several rounds over, so that machine load affects them alike; the table
shows the median over rounds of the median warm run. A run is everything from an 8-bit image in host
memory to keypoints, scores and descriptors in host memory.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE = os.path.join(ROOT, "third_party", "ALIKED", "assets", "st_pauls_cathedral", "1.jpg")


def resized(bgr, w, h):
    """Centre crop to the aspect ratio of w x h, then resize."""
    ih, iw = bgr.shape[:2]
    if iw * h > ih * w:
        cw = round(ih * w / h)
        bgr = bgr[:, (iw - cw) // 2:(iw - cw) // 2 + cw]
    else:
        ch = round(iw * h / w)
        bgr = bgr[(ih - ch) // 2:(ih - ch) // 2 + ch]
    interpolation = cv2.INTER_AREA if bgr.shape[1] >= w else cv2.INTER_CUBIC
    return cv2.resize(bgr, (w, h), interpolation=interpolation)


def jkp_column(binary, extra, args):
    def run(path, _bgr):
        cmd = [binary, "bench", path, "--runs", str(args.runs), "--top-k", str(args.top_k), "--model", args.model]
        cmd += extra.split()
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        return float(re.search(r"median ([0-9.]+) ms", out).group(1))
    return run


def torch_column(device, args):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import upstream

    if args.model == "xfeat":
        import xfeat_upstream
        model = xfeat_upstream.load_model(device, args.top_k)

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    if args.model != "xfeat":
        model = upstream.load_model(device=device, top_k=args.top_k)

    @torch.no_grad()
    def once_xfeat(rgb):
        out = model.detectAndCompute(upstream.to_tensor(rgb, device), top_k=args.top_k)[0]
        out = {k: v.cpu().numpy() for k, v in out.items()}
        sync()
        return out

    @torch.no_grad()
    def once(rgb):
        if args.model == "xfeat":
            return once_xfeat(rgb)
        image = upstream.to_tensor(rgb, model.device)
        feature_map, score_map = model.extract_dense_map(image)
        keypoints, scores, _ = model.dkd(score_map)
        descriptors, _ = model.desc_head(feature_map, keypoints)
        h, w = image.shape[-2:]
        wh = torch.tensor([w - 1, h - 1], device=image.device)
        out = ((wh * (keypoints[0] + 1) / 2).cpu().numpy(), scores[0].cpu().numpy(), descriptors[0].cpu().numpy())
        sync()
        return out

    def run(_path, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for _ in range(3):
            once(rgb)
        ms = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            once(rgb)
            ms.append(1e3 * (time.perf_counter() - t0))
        return statistics.median(ms)
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--sizes", default="640x480,1024x768,1920x1080")
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--model", default="aliked-n16", help="aliked-n16 or xfeat (also passed to jkp)")
    ap.add_argument("--runs", type=int, default=30, help="warm runs per measurement")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--binary", default=os.path.join(ROOT, "build", "release", "jkp"))
    ap.add_argument("--jkp", action="append", default=[], metavar="ARGS", help='jkp options, e.g. "--backend cuda"')
    ap.add_argument("--torch", action="append", default=[], metavar="DEVICE", help="original model on cpu, cuda or mps")
    args = ap.parse_args()

    columns = [("jkp " + re.sub(r"--(backend|precision) ", "", a), jkp_column(args.binary, a, args)) for a in args.jkp]
    columns += [("PyTorch " + d, torch_column(d, args)) for d in args.torch]
    if not columns:
        ap.error("nothing to time: give --jkp and/or --torch")
    source = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if source is None:
        sys.exit(f"cannot read image: {args.image}")

    print(f"top-k {args.top_k}, median of {args.runs} warm runs, median over {args.rounds} rounds, ms per image\n")
    print("| Size | " + " | ".join(name for name, _ in columns) + " |")
    print("|---|" + "---|" * len(columns))
    with tempfile.TemporaryDirectory() as tmp:
        for size in args.sizes.split(","):
            w, h = (int(v) for v in size.split("x"))
            bgr = resized(source, w, h)
            path = os.path.join(tmp, f"{size}.png")
            cv2.imwrite(path, bgr)
            times = [[] for _ in columns]
            for _ in range(args.rounds):
                for i, (_, run) in enumerate(columns):
                    times[i].append(run(path, bgr))
            print(f"| {size} | " + " | ".join(f"{statistics.median(t):.1f}" for t in times) + " |", flush=True)


if __name__ == "__main__":
    main()
