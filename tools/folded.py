"""ALIKED forward pass driven only by a .jkpw weights file (BatchNorm already folded).

This is the executable specification for the C++ backends: every step here is what the
native code has to reproduce. It uses torch only as a tensor library; nothing from the
original ALIKED repository is imported.
"""
import torch
import torch.nn.functional as F
import torchvision

import jkpfile

NMS_RADIUS = 2
DKD_RADIUS = 2
DKD_TEMPERATURE = 0.1


class Folded:
    def __init__(self, weights_path, device="cpu", top_k=-1, scores_th=0.2, n_limit=5000):
        self.meta, tensors = jkpfile.read(weights_path)
        self.w = {k: torch.from_numpy(v.copy()).float().to(device) for k, v in tensors.items()}
        self.device = device
        self.top_k, self.scores_th, self.n_limit = top_k, scores_th, n_limit
        self.K, self.M = int(self.meta["K"]), int(self.meta["M"])

    # ---------------------------------------------------------------- dense network
    def conv(self, x, name, padding):
        return F.conv2d(x, self.w[name + ".weight"], self.w.get(name + ".bias"), padding=padding)

    def dcn(self, x, name):
        """3x3 deformable convolution: offsets come from a regular 3x3 conv on the same input."""
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.0
        offset = self.conv(x, name + ".offset", 1).clamp(-max_offset, max_offset)
        return torchvision.ops.deform_conv2d(x, offset, self.w[name + ".weight"], self.w[name + ".bias"], padding=1)

    def block(self, x, name, deformable, residual):
        conv = self.dcn if deformable else (lambda t, n: self.conv(t, n, 1))
        y = F.selu(conv(x, name + ".conv1"))
        y = conv(y, name + ".conv2")
        if residual:
            y = y + self.conv(x, name + ".down", 0)
        return F.selu(y)

    def dense(self, image):
        """image: 1x3xHxW float in [0,1]. Returns (feature_map 1xCxHxW, score_map 1x1xHxW, stages)."""
        h, w = image.shape[2:]
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
        pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]  # left, right, top, bottom
        x = F.pad(image, pad, mode="replicate")

        x1 = self.block(x, "block1", False, False)
        x2 = self.block(F.avg_pool2d(x1, 2), "block2", False, True)
        x3 = self.block(F.avg_pool2d(x2, 4), "block3", True, True)
        x4 = self.block(F.avg_pool2d(x3, 4), "block4", True, True)

        a1 = F.selu(self.conv(x1, "conv1", 0))
        a2 = F.selu(self.conv(x2, "conv2", 0))
        a3 = F.selu(self.conv(x3, "conv3", 0))
        a4 = F.selu(self.conv(x4, "conv4", 0))

        def up(t, s):
            return F.interpolate(t, scale_factor=s, mode="bilinear", align_corners=True)

        x1234 = torch.cat([a1, up(a2, 2), up(a3, 8), up(a4, 32)], dim=1)

        s = F.selu(self.conv(x1234, "score_head.0", 0))
        s = F.selu(self.conv(s, "score_head.2", 1))
        s = F.selu(self.conv(s, "score_head.4", 1))
        score_map = torch.sigmoid(self.conv(s, "score_head.6", 1))
        feature_map = F.normalize(x1234, p=2, dim=1)

        crop = (slice(None), slice(None), slice(pad[2], pad[2] + h), slice(pad[0], pad[0] + w))
        stages = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "a1": a1, "a2": a2, "a3": a3, "a4": a4, "x1234": x1234}
        return feature_map[crop], score_map[crop], stages

    # ---------------------------------------------------------------- keypoint detection (DKD)
    @staticmethod
    def nms(scores):
        k, r = 2 * NMS_RADIUS + 1, NMS_RADIUS

        def mp(t):
            return F.max_pool2d(t, kernel_size=k, stride=1, padding=r)

        zeros = torch.zeros_like(scores)
        max_mask = scores == mp(scores)
        for _ in range(2):
            supp_mask = mp(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == mp(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    def detect(self, score_map):
        """score_map: 1x1xHxW. Returns (keypoints_norm Nx2 in [-1,1], scores N, dispersity N, nms_scores HxW)."""
        _, _, h, w = score_map.shape
        r = DKD_RADIUS
        nms = self.nms(score_map)
        nms[:, :, :r, :] = 0
        nms[:, :, :, :r] = 0
        nms[:, :, h - r:, :] = 0
        nms[:, :, :, w - r:] = 0
        flat_nms, flat_scores = nms.reshape(-1), score_map.reshape(-1)

        if self.top_k > 0:
            idx = torch.topk(flat_nms, self.top_k).indices
        else:
            mask = flat_nms > self.scores_th if self.scores_th > 0 else None
            if mask is None or mask.sum() == 0:
                mask = flat_nms > flat_scores.mean()
            idx = mask.nonzero()[:, 0]  # raster order
            if len(idx) > self.n_limit:
                idx = idx[flat_scores[idx].sort(descending=True)[1][:self.n_limit]]

        # Soft-argmax refinement over the (2r+1)x(2r+1) window; outside the map counts as score 0.
        ks = 2 * r + 1
        patches = F.unfold(score_map, kernel_size=ks, padding=r)[0].t()[idx]  # N x ks*ks, row-major (dy, dx)
        g = torch.linspace(-r, r, ks, device=score_map.device)
        grid = torch.stack([g.repeat(ks), g.repeat_interleave(ks)], dim=1)  # ks*ks x 2 as (dx, dy)
        x_exp = ((patches - patches.max(dim=1, keepdim=True).values) / DKD_TEMPERATURE).exp()
        residual = x_exp @ grid / x_exp.sum(dim=1, keepdim=True)
        dist2 = torch.norm((grid[None] - residual[:, None]) / r, dim=-1) ** 2
        dispersity = (x_exp * dist2).sum(dim=1) / x_exp.sum(dim=1)

        # Keypoints are stored normalized to [-1, 1]. The descriptor head truncates the pixel position
        # recovered from this normalized value, so the same float32 operation order must be kept.
        xy = torch.stack([idx % w, torch.div(idx, w, rounding_mode="trunc")], dim=1) + residual
        wh = torch.tensor([w - 1, h - 1], device=score_map.device)
        kpts_norm = xy / wh * 2 - 1
        scores = F.grid_sample(score_map, kpts_norm.view(1, 1, -1, 2), mode="bilinear", align_corners=True)[0, 0, 0]
        return kpts_norm, scores, dispersity, nms[0, 0]

    # ---------------------------------------------------------------- descriptor head (SDDH)
    def describe(self, feature_map, kpts_norm):
        """feature_map: 1xCxHxW (L2-normalized), kpts_norm: Nx2. Returns (descriptors NxD, offsets NxMx2)."""
        _, c, h, w = feature_map.shape
        n, K, M = len(kpts_norm), self.K, self.M
        wh = torch.tensor([[w - 1, h - 1]], device=feature_map.device)
        max_offset = max(h, w) / 4.0
        kpts = (kpts_norm / 2 + 0.5) * wh  # pixel xy, float32
        ij = kpts.long()  # truncation

        # KxK patch centred on the truncated keypoint, zeros outside the map.
        lo = (K - 1) // 2
        padded = F.pad(feature_map, (lo, K - 1 - lo, lo, K - 1 - lo))[0]
        k = torch.arange(K, device=feature_map.device)
        patch = padded[:, ij[:, 1, None, None] + k[None, :, None], ij[:, 0, None, None] + k[None, None, :]]
        patch = patch.permute(1, 0, 2, 3)  # N x C x K x K

        o = F.selu(F.conv2d(patch, self.w["desc_head.offset_conv.0.weight"], self.w["desc_head.offset_conv.0.bias"]))
        o = F.conv2d(o, self.w["desc_head.offset_conv.2.weight"], self.w["desc_head.offset_conv.2.bias"])
        offsets = o.clamp(-max_offset, max_offset)[:, :, 0, 0].view(n, 2, M).permute(0, 2, 1)  # N x M x (dx, dy)

        # Bilinear samples of the feature map at keypoint + offset (zeros outside the map).
        pos = 2.0 * (kpts.unsqueeze(1) + offsets) / wh[None] - 1
        f = F.grid_sample(feature_map, pos.reshape(1, n * M, 1, 2), mode="bilinear", align_corners=True)
        f = f.reshape(c, n, M, 1).permute(1, 0, 2, 3)  # N x C x M x 1
        f = F.selu(F.conv2d(f, self.w["desc_head.sf_conv.weight"])).squeeze(-1)  # N x C x M
        desc = torch.einsum("ncp,pcd->nd", f, self.w["desc_head.agg_weights"])
        return F.normalize(desc, p=2.0, dim=1), offsets

    @torch.no_grad()
    def describe_at(self, rgb, xy):
        """Descriptors at given keypoints, e.g. from another detector. rgb: HxWx3 uint8;
        xy: (N,2) pixels, (0, 0) = centre of the top-left pixel. Returns N x D."""
        image = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        feature_map, _, _ = self.dense(image)
        h, w = rgb.shape[:2]
        wh = torch.tensor([w - 1, h - 1], device=self.device)
        kpts_norm = 2.0 * torch.from_numpy(xy).to(self.device) / wh - 1
        return self.describe(feature_map, kpts_norm)[0].cpu().numpy()

    # ---------------------------------------------------------------- full pipeline
    @torch.no_grad()
    def run(self, rgb, stages=False):
        """rgb: HxWx3 uint8. Returns the same dict layout as upstream.run."""
        image = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        feature_map, score_map, st = self.dense(image)
        kpts_norm, scores, dispersity, nms = self.detect(score_map)
        desc, offsets = self.describe(feature_map, kpts_norm)
        h, w = rgb.shape[:2]
        wh = torch.tensor([w - 1, h - 1], device=self.device)
        out = {"keypoints": wh * (kpts_norm + 1) / 2, "scores": scores, "descriptors": desc}
        if stages:
            out.update({k: v[0] for k, v in st.items()})
            out.update({"score_map": score_map[0, 0], "feature_map": feature_map[0], "nms_scores": nms,
                        "keypoints_norm": kpts_norm, "score_dispersity": dispersity, "sddh_offsets": offsets})
        return {k: v.cpu().numpy() for k, v in out.items()}
