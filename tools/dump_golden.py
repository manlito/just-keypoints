#!/usr/bin/env python3
"""Dump golden test vectors from the original ALIKED: every intermediate tensor plus the final output.

    python tools/dump_golden.py third_party/ALIKED/assets/tum/*.png --max-size 320

Writes build/golden/<image>.jkpt (same container as .jkpw). Tensors: image, pad, x1..x4, a1..a4, x1234,
score_map, feature_map, nms_scores, keypoints_norm, keypoints, scores, score_dispersity, sddh_offsets, descriptors.
Files are large (feature maps at full resolution); they are generated locally, not committed.
"""
import argparse
import os

import jkpfile
import upstream


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="aliked-n16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-size", type=int, default=320, help="downscale so the longer side is at most this (0 = off)")
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--scores-th", type=float, default=0.2)
    ap.add_argument("--n-limit", type=int, default=5000)
    ap.add_argument("--out-dir", default=os.path.join(upstream.ROOT, "build", "golden"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = upstream.load_model(args.model, args.device, args.top_k, args.scores_th, args.n_limit)
    for path in args.images:
        rgb = upstream.load_image(path, args.max_size)
        out = upstream.run(model, rgb, stages=True)
        out["image_f32"] = rgb.astype("float32")  # the C++ reader only loads float32 tensors
        meta = {"kind": "golden", "model": args.model, "source_commit": upstream.PINNED_COMMIT,
                "image": os.path.basename(path), "width": rgb.shape[1], "height": rgb.shape[0],
                "top_k": args.top_k, "scores_th": args.scores_th, "n_limit": args.n_limit, "device": args.device}
        name = os.path.splitext(os.path.basename(path))[0]
        dest = os.path.join(args.out_dir, f"{name}.jkpt")
        jkpfile.write(dest, meta, out)
        print(f"{dest}: {rgb.shape[1]}x{rgb.shape[0]}, {len(out['keypoints'])} keypoints, "
              f"{os.path.getsize(dest) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
