#!/usr/bin/env python3
"""Compare two implementations of a model (ALIKED or RaCo) on the same images.

    python tools/compare.py IMAGES... [--a upstream] [--b folded]

Implementations (--a / --b):
    upstream[:cpu|cuda]   the original repository under PyTorch
    folded[:cpu|cuda]     reference forward pass from weights/<model>.jkpw (tools/folded.py, tools/raco_folded.py)
    cmd:<command>         external program, e.g. the jkp CLI. It is run as
                          <command> --image IN.png --out OUT.jkpt --top-k K --scores-th T --n-limit N
                          and must write keypoints (N,2 pixel xy), scores (N) and, when the model has them,
                          descriptors (N,D), ranker (N) and covariances (N,2,2).

--model raco: a detector without descriptors. --top-k is the number of keypoints (default 2048);
--scores-th and --n-limit do not apply. With --descriptor aliked-n16 the RaCo keypoints of each side are
described by that side's ALIKED (upstream: the original ALIKED; folded: tools/folded.py; cmd: pass
"--descriptor aliked-n16" in the command yourself).

Keypoints are paired by nearest position (within --match-px). Reported per image:
    n_a, n_b      keypoint counts
    overlap       fraction of keypoints paired, relative to the larger set
    pos           position error of pairs, in pixels (max)
    score         absolute score error of pairs (max); for RaCo relative to the best score
    desc          cosine similarity between paired descriptors (min)
    rank          ranker score error of pairs, relative to the largest score (max)  RaCo only
    cov           covariance error of pairs, relative Frobenius norm (max)         RaCo only
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

import jkpfile
import upstream


def with_descriptors(detect, describe):
    def run(rgb):
        out = detect(rgb)
        out["descriptors"] = describe(rgb, out["keypoints"])
        return out
    return run


def make_runner(spec, args):
    kind, _, rest = spec.partition(":")
    if args.model == "raco" and kind == "upstream":
        import raco_upstream
        model = raco_upstream.load_model(rest or "cpu", args.top_k)
        run = lambda rgb: raco_upstream.run(model, rgb)  # noqa: E731
        if args.descriptor:
            aliked = upstream.load_model(args.descriptor, rest or "cpu")
            run = with_descriptors(run, lambda rgb, xy: upstream.describe_at(aliked, rgb, xy))
        return run
    if args.model == "raco" and kind == "folded":
        from raco_folded import RacoFolded
        weights = args.weights or os.path.join(upstream.ROOT, "weights", "raco.jkpw")
        run = RacoFolded(weights, rest or "cpu", args.top_k).run
        if args.descriptor:
            from folded import Folded
            aliked = Folded(os.path.join(upstream.ROOT, "weights", f"{args.descriptor}.jkpw"), rest or "cpu")
            run = with_descriptors(run, aliked.describe_at)
        return run
    if kind == "upstream":
        model = upstream.load_model(args.model, rest or "cpu", args.top_k, args.scores_th, args.n_limit)
        return lambda rgb: upstream.run(model, rgb)
    if kind == "folded":
        from folded import Folded
        weights = args.weights or os.path.join(upstream.ROOT, "weights", f"{args.model}.jkpw")
        model = Folded(weights, rest or "cpu", args.top_k, args.scores_th, args.n_limit)
        return model.run
    if kind == "cmd":
        def run(rgb):
            with tempfile.TemporaryDirectory() as d:
                img, out = os.path.join(d, "in.png"), os.path.join(d, "out.jkpt")
                cv2.imwrite(img, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                subprocess.run(shlex.split(rest) + ["--image", img, "--out", out, "--top-k", str(args.top_k),
                                                    "--scores-th", str(args.scores_th), "--n-limit", str(args.n_limit)],
                               check=True)
                return {k: np.array(v) for k, v in jkpfile.read(out)[1].items()}
        return run
    sys.exit(f"unknown implementation: {spec}")


def compare(a, b, match_px):
    ka, kb = a["keypoints"].astype(np.float64), b["keypoints"].astype(np.float64)
    res = {"n_a": len(ka), "n_b": len(kb), "overlap": 0.0, "pos": float("nan"), "score": float("nan"),
           "desc": float("nan"), "rank": float("nan"), "cov": float("nan")}
    if len(ka) == 0 or len(kb) == 0:
        res["overlap"] = 1.0 if len(ka) == len(kb) else 0.0
        return res
    d = np.linalg.norm(ka[:, None] - kb[None], axis=2)
    j = d.argmin(axis=1)
    mutual = d.argmin(axis=0)[j] == np.arange(len(ka))
    ok = mutual & (d[np.arange(len(ka)), j] <= match_px)
    ia, ib = np.nonzero(ok)[0], j[ok]
    res["overlap"] = len(ia) / max(len(ka), len(kb))
    if len(ia):
        res["pos"] = float(d[ia, ib].max())
        res["score"] = float(np.abs(a["scores"][ia].astype(np.float64) - b["scores"][ib]).max())
        if "descriptors" in a and "descriptors" in b:
            da, db = a["descriptors"][ia].astype(np.float64), b["descriptors"][ib].astype(np.float64)
            cos = (da * db).sum(1) / (np.linalg.norm(da, axis=1) * np.linalg.norm(db, axis=1))
            res["desc"] = float(cos.min())
        if "ranker" in a and "ranker" in b:
            res["rank"] = float(np.abs(a["ranker"][ia].astype(np.float64) - b["ranker"][ib]).max())
        if "covariances" in a and "covariances" in b:
            ca = a["covariances"][ia].astype(np.float64).reshape(-1, 4)
            cb = b["covariances"][ib].astype(np.float64).reshape(-1, 4)
            res["cov"] = float((np.linalg.norm(ca - cb, axis=1) / np.linalg.norm(ca, axis=1)).max())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--a", default="upstream")
    ap.add_argument("--b", default="folded")
    ap.add_argument("--model", default="aliked-n16")
    ap.add_argument("--descriptor", default=None, help="with --model raco: describe the keypoints, e.g. aliked-n16")
    ap.add_argument("--weights", default=None, help=".jkpw file for 'folded' (default weights/<model>.jkpw)")
    ap.add_argument("--max-size", type=int, default=0, help="downscale so the longer side is at most this (0 = off)")
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--scores-th", type=float, default=0.2)
    ap.add_argument("--n-limit", type=int, default=5000)
    ap.add_argument("--match-px", type=float, default=0.5)
    ap.add_argument("--tol-pos", type=float, default=0.01, help="pass threshold: max position error in pixels")
    ap.add_argument("--tol-score", type=float, default=1e-3, help="pass threshold: max score error")
    ap.add_argument("--tol-desc", type=float, default=0.999, help="pass threshold: min descriptor cosine")
    ap.add_argument("--tol-overlap", type=float, default=0.999, help="pass threshold: min keypoint overlap")
    ap.add_argument("--tol-rank", type=float, default=1e-3, help="pass threshold: max relative ranker score error")
    ap.add_argument("--tol-cov", type=float, default=1e-3, help="pass threshold: max relative covariance error")
    args = ap.parse_args()

    run_a, run_b = make_runner(args.a, args), make_runner(args.b, args)
    print(f"a = {args.a}, b = {args.b}, model = {args.model}")
    raco = args.model == "raco"
    with_desc = not raco or bool(args.descriptor)
    extra = (f"{'rank':>9} {'cov':>9}" if raco else "") + (f"{'1-desc':>10}" if with_desc else "")
    print(f"{'image':28} {'size':>10} {'n_a':>6} {'n_b':>6} {'overlap':>8} {'pos px':>9} {'score':>9} "
          f"{extra} {'t_a s':>7} {'t_b s':>7}  result")
    failed = 0
    for path in args.images:
        rgb = upstream.load_image(path, args.max_size)
        t0 = time.perf_counter()
        a = run_a(rgb)
        t1 = time.perf_counter()
        b = run_b(rgb)
        t2 = time.perf_counter()
        r = compare(a, b, args.match_px)
        if raco:
            # Scores are softmax probabilities (they sum to 1 over the image): compare them relative to the best.
            # Ranker scores are unbounded (thousands on a typical image): compare them relative to the largest.
            r["score"] /= max(float(a["scores"].max()), 1e-30) if r["n_a"] else 1.0
            r["rank"] /= max(float(np.abs(a["ranker"]).max()), 1e-30) if r["n_a"] else 1.0
            ok_extra = r["rank"] <= args.tol_rank and r["cov"] <= args.tol_cov
            extra = f"{r['rank']:9.2e} {r['cov']:9.2e}"
        else:
            ok_extra, extra = True, ""
        if with_desc:
            ok_extra = ok_extra and r["desc"] >= args.tol_desc
            extra += f"{1 - r['desc']:10.2e}"
        ok = (r["overlap"] >= args.tol_overlap and
              (r["n_a"] == 0 or (r["pos"] <= args.tol_pos and r["score"] <= args.tol_score and ok_extra)))
        failed += not ok
        size = f"{rgb.shape[1]}x{rgb.shape[0]}"
        print(f"{os.path.basename(path)[:28]:28} {size:>10} {r['n_a']:6d} {r['n_b']:6d} {r['overlap']:8.4f} "
              f"{r['pos']:9.2e} {r['score']:9.2e} {extra} {t1 - t0:7.2f} {t2 - t1:7.2f}  "
              f"{'ok' if ok else 'FAIL'}")
    print(f"{len(args.images) - failed}/{len(args.images)} images within tolerance")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
