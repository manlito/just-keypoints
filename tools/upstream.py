"""Run the original ALIKED implementation (PyTorch, CPU or CUDA) and expose its intermediate tensors.

The original code is not vendored: tools/fetch_upstream.sh clones it at the pinned commit
into third_party/ALIKED (override the location with JKP_ALIKED_DIR).
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIKED_DIR = os.environ.get("JKP_ALIKED_DIR", os.path.join(ROOT, "third_party", "ALIKED"))
PINNED_COMMIT = "683d7c65197395c0b3f01ebe76e1084a27e73a65"


def get_patches_cpu(fmap, points, kernel_size):
    """Pure-torch equivalent of the original compiled custom op (custom_ops/get_patches.cpp):
    zero-pad the map and take a KxK patch centred on each integer point. fmap: CxHxW, points: Nx2 (x, y)."""
    lo = (kernel_size - 1) // 2
    hi = kernel_size - 1 - lo
    padded = F.pad(fmap.unsqueeze(0), (lo, hi, lo, hi)).squeeze(0)
    k = torch.arange(kernel_size, device=fmap.device)
    ys = points[:, 1, None, None] + k[None, :, None]  # N x K x 1
    xs = points[:, 0, None, None] + k[None, None, :]  # N x 1 x K
    return padded[:, ys, xs].permute(1, 0, 2, 3)  # N x C x K x K


def load_model(model_name="aliked-n16", device="cpu", top_k=-1, scores_th=0.2, n_limit=5000):
    if not os.path.isdir(ALIKED_DIR):
        sys.exit(f"Original ALIKED not found at {ALIKED_DIR}. Run tools/fetch_upstream.sh first.")
    if ALIKED_DIR not in sys.path:
        sys.path.insert(0, ALIKED_DIR)
    from nets.aliked import ALIKED  # noqa: E402

    model = ALIKED(model_name=model_name, device=device, top_k=top_k, scores_th=scores_th, n_limit=n_limit)
    # The original needs a compiled CUDA extension for get_patches. When it has not been built,
    # use the equivalent pure-torch version so the original can run on CPU-only machines.
    if not glob.glob(os.path.join(ALIKED_DIR, "custom_ops", "get_patches*.so")):
        model.desc_head.get_patches_func = get_patches_cpu
    return model


@torch.no_grad()
def describe_at(model, rgb, xy):
    """Descriptors of the original ALIKED at given keypoints. xy: (N,2) pixels, (0, 0) = centre of the
    top-left pixel. Same steps as ALIKED.describe in cvg/LightGlue, which is how the RaCo authors pair
    RaCo keypoints with ALIKED descriptors (their keypoints carry a +0.5 offset that it removes first)."""
    image = to_tensor(rgb, model.device)
    h, w = image.shape[-2:]
    wh = torch.tensor([w - 1, h - 1], device=image.device)
    keypoints = torch.from_numpy(xy).to(image.device) + 0.5
    keypoints_n = 2.0 * (keypoints - 0.5) / wh[None] - 1
    feature_map, _ = model.extract_dense_map(image)
    return model.desc_head(feature_map, [keypoints_n])[0][0].cpu().numpy()


def load_image(path, max_size=0):
    """Returns an HxWx3 uint8 RGB image, optionally downscaled so its longer side is <= max_size."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"cannot read image: {path}")
    h, w = bgr.shape[:2]
    if max_size and max(h, w) > max_size:
        s = max_size / max(h, w)
        bgr = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def to_tensor(rgb, device="cpu"):
    """HxWx3 uint8 -> 1x3xHxW float32 in [0, 1] (same as torchvision ToTensor)."""
    return torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


@torch.no_grad()
def run(model, rgb, stages=False):
    """Run the original model on one image. Returns a dict of numpy arrays.

    Always: keypoints (N,2 pixel xy), scores (N), descriptors (N,D).
    With stages=True: every intermediate tensor as well (used for golden test vectors)."""
    from nets.padder import InputPadder
    from nets.soft_detect import simple_nms

    m = model
    image = to_tensor(rgb, m.device)
    out = {}

    # Same statements as ALIKED.extract_dense_map, keeping the intermediates.
    padder = InputPadder(image.shape[-2], image.shape[-1], 32)
    padded = padder.pad(image)
    x1 = m.block1(padded)
    x2 = m.block2(m.pool2(x1))
    x3 = m.block3(m.pool4(x2))
    x4 = m.block4(m.pool4(x3))
    a1 = m.gate(m.conv1(x1))
    a2 = m.gate(m.conv2(x2))
    a3 = m.gate(m.conv3(x3))
    a4 = m.gate(m.conv4(x4))
    x1234 = torch.cat([a1, m.upsample2(a2), m.upsample8(a3), m.upsample32(a4)], dim=1)
    score_map = padder.unpad(torch.sigmoid(m.score_head(x1234.clone())))
    feature_map = padder.unpad(F.normalize(x1234, p=2, dim=1))

    # Guard: the staged copy above must be bit-identical to the original function.
    ref_feature_map, ref_score_map = m.extract_dense_map(image)
    assert torch.equal(feature_map, ref_feature_map) and torch.equal(score_map, ref_score_map)

    keypoints, kptscores, dispersity = m.dkd(score_map)
    descriptors, offsets = m.desc_head(feature_map, keypoints)

    _, _, h, w = image.shape
    wh = torch.tensor([w - 1, h - 1], device=image.device)
    out["keypoints"] = wh * (keypoints[0] + 1) / 2  # same conversion as ALIKED.run
    out["scores"] = kptscores[0]
    out["descriptors"] = descriptors[0]

    if stages:
        nms = simple_nms(score_map, 2)
        r = m.dkd.radius
        nms[:, :, :r, :] = 0
        nms[:, :, :, :r] = 0
        nms[:, :, h - r:, :] = 0
        nms[:, :, :, w - r:] = 0
        out.update({
            "image": torch.from_numpy(rgb),
            "pad": torch.tensor(padder._pad, dtype=torch.int32),  # left, right, top, bottom
            "x1": x1[0], "x2": x2[0], "x3": x3[0], "x4": x4[0],
            "a1": a1[0], "a2": a2[0], "a3": a3[0], "a4": a4[0],
            "x1234": x1234[0],
            "score_map": score_map[0, 0],
            "feature_map": feature_map[0],
            "nms_scores": nms[0, 0],
            "keypoints_norm": keypoints[0],
            "score_dispersity": dispersity[0],
            "sddh_offsets": offsets[0],
        })
    return {k: v.detach().cpu().numpy() for k, v in out.items()}
