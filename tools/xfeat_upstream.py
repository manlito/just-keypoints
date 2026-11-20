"""Run the original XFeat implementation (PyTorch, CPU or CUDA) and expose its intermediate tensors.

The original code is not vendored: tools/fetch_upstream.sh clones verlab/accelerated_features at the pinned
commit into third_party/accelerated_features, checkpoint included (override with JKP_XFEAT_DIR).

Sparse mode only (XFeat.detectAndCompute). Keypoints are returned in the jkp convention: (0, 0) is the centre of
the top-left pixel. The original resizes the image to a multiple of 32 and returns the resized image's pixel i
as i * r (r = original size / resized size); here it is (i + 0.5) * r - 0.5, the same pixel centre in the
original image. That is the only difference; the order of equal scores is left as the original's.
"""
import contextlib
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from upstream import ROOT, load_image, to_tensor  # noqa: F401  (re-exported for the other tools)

XFEAT_DIR = os.environ.get("JKP_XFEAT_DIR", os.path.join(ROOT, "third_party", "accelerated_features"))
PINNED_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
DEFAULT_TOP_K = 4096
DETECTION_THRESHOLD = 0.05


def load_model(device="cpu", top_k=DEFAULT_TOP_K):
    if not os.path.isfile(os.path.join(XFEAT_DIR, "weights", "xfeat.pt")):
        sys.exit(f"Original XFeat not found at {XFEAT_DIR}. Run tools/fetch_upstream.sh first.")
    if XFEAT_DIR not in sys.path:
        sys.path.insert(0, XFEAT_DIR)
    from modules.xfeat import XFeat  # noqa: E402

    with contextlib.redirect_stdout(io.StringIO()):  # the original prints a "loading weights" line
        model = XFeat(weights=os.path.join(XFEAT_DIR, "weights", "xfeat.pt"),
                      top_k=top_k if top_k > 0 else DEFAULT_TOP_K, detection_threshold=DETECTION_THRESHOLD)
    model.dev = torch.device(device)
    return model.to(device).eval()


@torch.no_grad()
def run(model, rgb, stages=False):
    """Run the original model on one image. Returns a dict of numpy arrays:
    keypoints (N,2 pixel xy), scores (N), descriptors (N,64); with stages=True every intermediate as well."""
    from modules.interpolator import InterpolateSparse2d

    m, net = model, model.net
    image = to_tensor(rgb, m.dev)
    h, w = image.shape[-2:]
    ref = m.detectAndCompute(image, top_k=m.top_k)[0]

    # Same statements as XFeat.detectAndCompute and XFeatModel.forward, keeping the intermediates.
    x, rh, rw = m.preprocess_tensor(image)
    _, _, H, W = x.shape
    g = x.mean(dim=1, keepdim=True)
    xn = net.norm(g)
    x1 = net.block1(xn)
    skip = net.skip1(xn)
    x2 = net.block2(x1 + skip)
    x3 = net.block3(x2)
    x4 = net.block4(x3)
    x5 = net.block5(x4)
    x4u = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
    x5u = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
    feats = net.block_fusion(x3 + x4u + x5u)
    heatmap = net.heatmap_head(feats)
    unfolded = net._unfold2d(xn, ws=8)
    logits = net.keypoint_head(unfolded)
    feats_n = F.normalize(feats, dim=1)
    kpts_heat = m.get_kpts_heatmap(logits)
    mkpts = m.NMS(kpts_heat, threshold=m.detection_threshold, kernel_size=5)
    scores = (InterpolateSparse2d("nearest")(kpts_heat, mkpts, H, W) *
              InterpolateSparse2d("bilinear")(heatmap, mkpts, H, W)).squeeze(-1)
    scores[torch.all(mkpts == 0, dim=-1)] = -1
    idxs = torch.argsort(-scores)
    pix = torch.gather(mkpts, 1, idxs[..., None].expand(-1, -1, 2))[:, :m.top_k]
    scores = torch.gather(scores, -1, idxs)[:, :m.top_k]
    desc = F.normalize(m.interpolator(feats_n, pix, H=H, W=W), dim=-1)
    valid = scores[0] > 0
    pix, scores, desc = pix[0][valid], scores[0][valid], desc[0][valid]

    # The staged copy must be the original, bit for bit.
    r = torch.tensor([rw, rh], device=pix.device)
    assert torch.equal(pix * r.view(1, -1), ref["keypoints"]), "staged copy differs from detectAndCompute"
    assert torch.equal(scores, ref["scores"]) and torch.equal(desc, ref["descriptors"])

    xy = (pix.double() + 0.5) * torch.tensor([rw, rh], dtype=torch.float64, device=pix.device) - 0.5
    out = {"keypoints": xy.float().cpu().numpy(), "scores": scores.cpu().numpy(),
           "descriptors": desc.cpu().numpy()}
    if stages:
        st = {"image": image[0], "resized": x[0], "gray": g[0], "norm": xn[0], "x1": x1[0], "skip": skip[0],
              "x2": x2[0], "x3": x3[0], "x4": x4[0], "x5": x5[0], "x4u": x4u[0], "x5u": x5u[0], "feats": feats[0],
              "heatmap": heatmap[0], "unfolded": unfolded[0], "logits": logits[0], "feats_n": feats_n[0],
              "kpts_heat": kpts_heat[0]}
        out.update({k: v.contiguous().cpu().numpy() for k, v in st.items()})
    return out
