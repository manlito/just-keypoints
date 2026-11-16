#!/usr/bin/env python3
"""Export an original ALIKED checkpoint to a .jkpw file with BatchNorm folded into the convolutions.

    python tools/export_weights.py                       # aliked-n16 -> weights/aliked-n16.jkpw
    python tools/export_weights.py --model aliked-t16
    python tools/export_weights.py --model raco          # original RaCo -> weights/raco.jkpw
"""
import argparse
import os
import subprocess

import torch

import jkpfile
import upstream


def fold(conv_weight, bn, prefix, sd):
    """conv (no bias) followed by BatchNorm in eval mode -> conv with weight and bias."""
    g, b = sd[f"{prefix}.{bn}.weight"], sd[f"{prefix}.{bn}.bias"]
    mean, var = sd[f"{prefix}.{bn}.running_mean"], sd[f"{prefix}.{bn}.running_var"]
    scale = g.double() / torch.sqrt(var.double() + 1e-5)  # nn.BatchNorm2d default eps
    weight = conv_weight.double() * scale[:, None, None, None]
    bias = b.double() - mean.double() * scale
    return weight.float(), bias.float()


def export(model_name):
    model = upstream.load_model(model_name)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out, used = {}, set()

    def take(key):
        used.add(key)
        return sd[key]

    for block in ("block1", "block2", "block3", "block4"):
        deformable = f"{block}.conv1.regular_conv.weight" in sd
        for i in ("1", "2"):
            conv, bn = f"{block}.conv{i}", f"bn{i}"
            if deformable:
                out[f"{conv}.offset.weight"] = take(f"{conv}.offset_conv.weight")
                out[f"{conv}.offset.bias"] = take(f"{conv}.offset_conv.bias")
                w = take(f"{conv}.regular_conv.weight")
            else:
                w = take(f"{conv}.weight")
            out[f"{conv}.weight"], out[f"{conv}.bias"] = fold(w, bn, block, sd)
            used.update(f"{block}.{bn}.{s}" for s in
                        ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"))
        if f"{block}.downsample.weight" in sd:
            out[f"{block}.down.weight"] = take(f"{block}.downsample.weight")
            out[f"{block}.down.bias"] = take(f"{block}.downsample.bias")
    for key in sd:
        if key.startswith(("conv", "score_head.", "desc_head.")):
            out[key] = take(key)
    missing = set(sd) - used
    assert not missing, f"checkpoint tensors not exported: {sorted(missing)}"

    from nets.aliked import ALIKED_CFGS
    commit = subprocess.run(["git", "-C", upstream.ALIKED_DIR, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    meta = {"kind": "weights", "model": model_name, **ALIKED_CFGS[model_name],
            "source": "https://github.com/Shiaoming/ALIKED", "source_commit": commit,
            "license": "BSD-3-Clause, Copyright (c) 2022, Zhao Xiaoming"}
    return meta, {k: v.numpy() for k, v in out.items()}


def export_raco():
    import numpy as np

    import raco_upstream

    model = raco_upstream.load_model()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out, used = {}, set()

    def take(key):
        used.add(key)
        return sd[key]

    def folded_pair(src, dst):
        """Two conv + BatchNorm pairs of a ConvBlock / ResBlock."""
        for i in ("1", "2"):
            out[f"{dst}.conv{i}.weight"], out[f"{dst}.conv{i}.bias"] = fold(take(f"{src}.conv{i}.weight"), f"bn{i}",
                                                                           src, sd)
            used.update(f"{src}.bn{i}.{s}" for s in
                        ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"))
        if f"{src}.match_dims.weight" in sd:
            out[f"{dst}.down.weight"] = take(f"{src}.match_dims.weight")
            out[f"{dst}.down.bias"] = take(f"{src}.match_dims.bias")

    out["norm.mean"] = np.array(model.normalizer.mean, dtype=np.float32)
    out["norm.std"] = np.array(model.normalizer.std, dtype=np.float32)
    for block in ("block1", "block2", "block3", "block4"):
        folded_pair(block, block)
    for key in sd:
        if key.startswith(("conv", "score_head.")):
            out[key] = take(key)
    blocks = len(model.ranker_head) - 1
    for i in range(blocks):
        folded_pair(f"ranker_head.{i}", f"ranker.{i}")
    out["ranker.out.weight"] = take(f"ranker_head.{blocks}.weight")
    out["ranker.out.bias"] = take(f"ranker_head.{blocks}.bias")
    for i in range(3):
        out[f"cov.{i}.weight"] = take(f"covariance_estimator_head.{2 * i}.weight")
    out["cov.3.weight"] = take("covariance_estimator_head.6.weight")
    out["cov.3.bias"] = take("covariance_estimator_head.6.bias")
    missing = set(sd) - used
    assert not missing, f"checkpoint tensors not exported: {sorted(missing)}"

    commit = subprocess.run(["git", "-C", raco_upstream.RACO_DIR, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    meta = {"kind": "weights", "model": "raco", "c1": 16, "c2": 32, "c3": 64, "c4": 128, "dim": 0,
            "ranker_blocks": blocks, "nms": model.conf.nms_radius, "subpixel_temp": model.conf.subpixel_temp,
            "top_k": model.default_conf["max_num_keypoints"],
            "source": "https://github.com/cvg/RaCo", "source_commit": commit,
            "source_weights": model.default_conf["weights"], "license": "Apache-2.0"}
    return meta, {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="aliked-n16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = args.out or os.path.join(upstream.ROOT, "weights", f"{args.model}.jkpw")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta, tensors = export_raco() if args.model == "raco" else export(args.model)
    jkpfile.write(path, meta, tensors)

    # Round-trip check.
    meta2, tensors2 = jkpfile.read(path)
    assert meta2 == {k: str(v) for k, v in meta.items()}
    assert list(tensors2) == list(tensors) and all((tensors2[k] == tensors[k]).all() for k in tensors)
    n = sum(t.size for t in tensors.values())
    print(f"wrote {path}: {len(tensors)} tensors, {n} values, {os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
