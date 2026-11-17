"""Run the original RaCo implementation (PyTorch, CPU or CUDA) and expose its intermediate tensors.

The original code is not vendored: tools/fetch_upstream.sh clones it at the pinned commit into
third_party/RaCo and downloads the released checkpoint to third_party/raco.pth
(override with JKP_RACO_DIR / JKP_RACO_WEIGHTS).

Keypoints are returned in the jkp convention: (0, 0) is the centre of the top-left pixel.
The original adds 0.5 (pixel-corner convention); that is the only difference.
"""
import contextlib
import io
import os
import sys

import torch
import torch.nn.functional as F

from upstream import ROOT, load_image, to_tensor  # noqa: F401  (re-exported for the other tools)

RACO_DIR = os.environ.get("JKP_RACO_DIR", os.path.join(ROOT, "third_party", "RaCo"))
RACO_WEIGHTS = os.environ.get("JKP_RACO_WEIGHTS", os.path.join(ROOT, "third_party", "raco.pth"))
PINNED_COMMIT = "ee06fa68dbb20303f334fbc257dad71d4149603b"
DEFAULT_TOP_K = 2048


def load_model(device="cpu", top_k=DEFAULT_TOP_K):
    if not os.path.isdir(RACO_DIR) or not os.path.isfile(RACO_WEIGHTS):
        sys.exit(f"Original RaCo not found at {RACO_DIR} / {RACO_WEIGHTS}. Run tools/fetch_upstream.sh first.")
    if RACO_DIR not in sys.path:
        sys.path.insert(0, RACO_DIR)
    from raco import RaCo  # noqa: E402

    with contextlib.redirect_stdout(io.StringIO()):  # the original prints a "loaded weights" line
        model = RaCo(weights=RACO_WEIGHTS, max_num_keypoints=top_k if top_k > 0 else DEFAULT_TOP_K)
    return model.eval().to(device)


@torch.no_grad()
def run(model, rgb, stages=False):
    """Run the original model on one image. Returns a dict of numpy arrays.

    Always: keypoints (N,2 pixel xy), scores (N), ranker (N), covariances (N,2,2).
    With stages=True: every intermediate tensor as well (used for golden test vectors)."""
    from raco.raco import InputPadder, _compute_subpixel_offsets, _sample_at_keypoints

    m = model
    device = next(m.parameters()).device
    image = to_tensor(rgb, device)
    h, w = image.shape[-2:]
    # The original always returns max_num_keypoints entries; an image has at most h*w candidates.
    k_saved = m.conf.max_num_keypoints
    m.conf.max_num_keypoints = min(k_saved, h * w)
    try:
        ref = m.forward({"image": image})

        # Same statements as RaCo.forward, keeping the intermediates.
        normalized = m.normalizer(image)
        padder = InputPadder(h, w, 32)
        x = padder.pad(normalized)
        x1 = m.block1(x)
        x2 = m.block2(m.pool2(x1))
        x3 = m.block3(m.pool4(x2))
        x4 = m.block4(m.pool4(x3))
        a1 = m.gate(m.conv1(x1))
        a2 = m.gate(m.conv2(x2))
        a3 = m.gate(m.conv3(x3))
        a4 = m.gate(m.conv4(x4))
        x1234 = torch.cat([a1, m.upsample2(a2), m.upsample8(a3), m.upsample32(a4)], dim=1)
        logits = padder.unpad(m.score_head(x1234))
        ranker_map = padder.unpad(m.ranker_head(x))
        chol = padder.unpad(m.covariance_estimator_head(x1234))
        chol = torch.stack([m.var_activation(chol[:, 0]), chol[:, 1], m.var_activation(chol[:, 2])], dim=1)
        probs = F.softmax(logits.flatten(1), dim=1).reshape(logits.size())

        r = m.conf.nms_radius
        nms = probs * (probs == F.max_pool2d(probs, r, stride=1, padding=r // 2))
        idx = torch.topk(nms.reshape(1, h * w), k=m.conf.max_num_keypoints).indices
        kpts = torch.stack([(idx % w).float(), (idx // w).float()], dim=-1)
        kpts = kpts + _compute_subpixel_offsets(logits, idx, r, m.conf.subpixel_temp)
        scores = _sample_at_keypoints(probs, kpts, h, w, True)
        ranker = _sample_at_keypoints(ranker_map, kpts, h, w, True)
        chol_kpts = _sample_at_keypoints(chol, kpts, h, w, True)
    finally:
        m.conf.max_num_keypoints = k_saved

    # Guard: the staged copy above must be bit-identical to the original function.
    assert torch.equal(kpts + 0.5, ref["keypoints"]) and torch.equal(scores, ref["keypoint_scores"])
    assert torch.equal(ranker, ref["ranker_scores"])

    # Entries with a suppressed score (0 after NMS) are filler: the original pads its output to
    # max_num_keypoints when an image has fewer local maxima. They are not keypoints.
    keep = nms.reshape(-1)[idx[0]] > 0
    out = {"keypoints": kpts[0][keep], "scores": scores[0][keep], "ranker": ranker[0][keep],
           "covariances": ref["covariances"][0][keep]}
    if stages:
        out.update({
            "image": torch.from_numpy(rgb),
            "pad": torch.tensor(padder._pad, dtype=torch.int32),  # left, right, top, bottom
            "x": x[0], "x1": x1[0], "x2": x2[0], "x3": x3[0], "x4": x4[0],
            "a1": a1[0], "a2": a2[0], "a3": a3[0], "a4": a4[0], "x1234": x1234[0],
            "logits": logits[0, 0], "probs": probs[0, 0], "nms_probs": nms[0, 0],
            "ranker_map": ranker_map[0, 0], "cholesky_map": chol[0],
            "cholesky": chol_kpts[0][keep],
        })
    return {k: v.detach().cpu().numpy() for k, v in out.items()}
