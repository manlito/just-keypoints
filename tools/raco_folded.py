"""RaCo forward pass driven only by a .jkpw weights file (BatchNorm already folded).

This is the executable specification for the C++ backends: every step here is what the
native code has to reproduce. It uses torch only as a tensor library; nothing from the
original RaCo repository is imported.

Differences from the original, all deliberate:
  * keypoints use the jkp convention, (0, 0) = centre of the top-left pixel (the original adds 0.5);
  * selection is deterministic: best probability first, ties in raster order (torch.topk leaves ties open);
  * suppressed pixels are never returned, so fewer than top_k keypoints come back when the image has
    fewer local maxima (the original pads its output with arbitrary pixels).
"""
import torch
import torch.nn.functional as F

import jkpfile

DEFAULT_TOP_K = 2048
LEAKY_SLOPE = 0.01  # nn.LeakyReLU default


class RacoFolded:
    def __init__(self, weights_path, device="cpu", top_k=DEFAULT_TOP_K):
        self.meta, tensors = jkpfile.read(weights_path)
        self.w = {k: torch.from_numpy(v.copy()).float().to(device) for k, v in tensors.items()}
        self.device = device
        self.top_k = top_k if top_k > 0 else DEFAULT_TOP_K
        self.nms = int(self.meta["nms"])                    # NMS and refinement window (3 = 3x3)
        self.temperature = float(self.meta["subpixel_temp"])
        self.ranker_blocks = int(self.meta["ranker_blocks"])

    # ---------------------------------------------------------------- dense network
    def conv(self, x, name, reflect=False):
        """Stride 1, "same" padding: zeros, or mirrored without repeating the border when reflect is set."""
        weight = self.w[name + ".weight"]
        p = weight.shape[-1] // 2
        if reflect and p:
            x, p = F.pad(x, (p, p, p, p), mode="reflect"), 0
        return F.conv2d(x, weight, self.w.get(name + ".bias"), padding=p)

    def conv_block(self, x, name):
        return F.selu(self.conv(F.selu(self.conv(x, name + ".conv1")), name + ".conv2"))

    def res_block(self, x, name):
        y = self.conv(F.selu(self.conv(x, name + ".conv1")), name + ".conv2")
        return F.selu(y + self.conv(x, name + ".down"))

    def dense(self, image):
        """image: 1x3xHxW float in [0,1]. Returns (logits, ranker_map, cholesky_map, stages), cropped to HxW."""
        h, w = image.shape[2:]
        x = (image - self.w["norm.mean"].view(1, 3, 1, 1)) / self.w["norm.std"].view(1, 3, 1, 1)
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
        pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]  # left, right, top, bottom
        x = F.pad(x, pad, mode="replicate")

        # Encoder: ALIKED-n16 widths, plain 3x3 convolutions.
        x1 = self.conv_block(x, "block1")
        x2 = self.res_block(F.avg_pool2d(x1, 2), "block2")
        x3 = self.res_block(F.avg_pool2d(x2, 4), "block3")
        x4 = self.res_block(F.avg_pool2d(x3, 4), "block4")

        a1 = F.selu(self.conv(x1, "conv1"))  # 1x1
        a2 = F.selu(self.conv(x2, "conv2"))  # 3x3
        a3 = F.selu(self.conv(x3, "conv3"))
        a4 = F.selu(self.conv(x4, "conv4"))

        def up(t, s):
            return F.interpolate(t, scale_factor=s, mode="bilinear", align_corners=True)

        x1234 = torch.cat([a1, up(a2, 2), up(a3, 8), up(a4, 32)], dim=1)

        # Score head: raw logits, no sigmoid.
        s = F.selu(self.conv(x1234, "score_head.0"))
        s = F.selu(self.conv(s, "score_head.2"))
        s = F.selu(self.conv(s, "score_head.4"))
        logits = self.conv(s, "score_head.6")

        # Ranker head: on the normalized, padded image at full resolution.
        r = x
        for i in range(self.ranker_blocks):
            r = self.res_block(r, f"ranker.{i}")
        ranker_map = self.conv(r, "ranker.out", reflect=True)  # 5x5

        # Covariance head: Cholesky elements (L11, L21, L22) per pixel; softplus keeps the diagonal positive.
        c = x1234
        for i in range(3):
            c = F.leaky_relu(self.conv(c, f"cov.{i}", reflect=True), LEAKY_SLOPE)
        c = self.conv(c, "cov.3")  # 1x1

        crop = (slice(None), slice(None), slice(pad[2], pad[2] + h), slice(pad[0], pad[0] + w))
        c = c[crop]
        cholesky_map = torch.stack([F.softplus(c[:, 0]), c[:, 1], F.softplus(c[:, 2])], dim=1)
        stages = {"x": x, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "a1": a1, "a2": a2, "a3": a3, "a4": a4,
                  "x1234": x1234}
        return logits[crop], ranker_map[crop], cholesky_map, stages

    # ---------------------------------------------------------------- keypoint detection
    def detect(self, logits):
        """logits: 1x1xHxW. Returns (xy Nx2 pixels, flat index N, probs 1x1xHxW, nms_probs HxW)."""
        _, _, h, w = logits.shape
        k, r = self.nms, self.nms // 2
        probs = F.softmax(logits.flatten(1), dim=1).reshape(logits.shape)  # over the whole image
        nms = probs * (probs == F.max_pool2d(probs, k, stride=1, padding=r))
        flat = nms.reshape(-1)
        order = torch.sort(flat, descending=True, stable=True).indices[:min(self.top_k, h * w)]
        idx = order[flat[order] > 0]

        # Subpixel refinement: softmax of the logits in the kxk window (outside the map counts as logit 0),
        # expected offset in pixels.
        patches = F.unfold(logits, kernel_size=k, padding=r)[0].t()[idx]  # N x k*k, row-major (dy, dx)
        g = torch.linspace(-r, r, k, device=logits.device)
        grid = torch.stack([g.repeat(k), g.repeat_interleave(k)], dim=1)  # k*k x 2 as (dx, dy)
        offset = (patches / self.temperature).softmax(dim=1) @ grid
        xy = torch.stack([idx % w, torch.div(idx, w, rounding_mode="trunc")], dim=1).float() + offset
        return xy, idx, probs, nms[0, 0]

    @staticmethod
    def sample(fmap, xy):
        """Bilinear samples of a 1xCxHxW map at pixel positions xy (N,2); positions are clamped to the map."""
        _, _, h, w = fmap.shape
        grid = torch.stack([2.0 * xy[:, 0] / (w - 1) - 1.0, 2.0 * xy[:, 1] / (h - 1) - 1.0], dim=-1)
        out = F.grid_sample(fmap, grid.view(1, -1, 1, 2), mode="bilinear", padding_mode="border",
                            align_corners=True)
        return out[0, :, :, 0].t()  # N x C

    # ---------------------------------------------------------------- full pipeline
    @torch.no_grad()
    def run(self, rgb, stages=False):
        """rgb: HxWx3 uint8. Returns the same dict layout as raco_upstream.run."""
        image = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        logits, ranker_map, cholesky_map, st = self.dense(image)
        xy, _, probs, nms = self.detect(logits)
        chol = self.sample(cholesky_map, xy)
        l11, l21, l22 = chol[:, 0], chol[:, 1], chol[:, 2]
        # Sigma = L L^T with L = [[L11, 0], [L21, L22]]
        cov = torch.stack([l11 * l11, l11 * l21, l11 * l21, l21 * l21 + l22 * l22], dim=1).view(-1, 2, 2)
        out = {"keypoints": xy, "scores": self.sample(probs, xy)[:, 0], "ranker": self.sample(ranker_map, xy)[:, 0],
               "covariances": cov}
        if stages:
            out.update({k: v[0] for k, v in st.items()})
            out.update({"logits": logits[0, 0], "probs": probs[0, 0], "nms_probs": nms,
                        "ranker_map": ranker_map[0, 0], "cholesky_map": cholesky_map[0], "cholesky": chol})
        return {k: v.cpu().numpy() for k, v in out.items()}
