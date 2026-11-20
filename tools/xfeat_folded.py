"""XFeat forward pass (sparse mode) driven only by a .jkpw weights file (BatchNorm already folded).

This is the executable specification for the C++ backends: every step here is what the native code has to
reproduce. It uses torch only as a tensor library; nothing from the original repository is imported.

Steps that the original leaves to torch.nn.functional.grid_sample are written out, because their details
decide which keypoints survive:
  * positions p (pixels of the resized image, H x W) are normalized as g = 2 * (p / (W - 1)) - 1 in float32
    and sampled with align_corners = False, so a map of width w is read at ((g + 1) * w - 1) / 2;
  * the keypoint heatmap (full resolution) is read with "nearest" (round half to even): the last row and the
    last column read outside the map, get 0 and are dropped, as in the original;
  * a keypoint at pixel (0, 0) is dropped too (the original marks its padding entries that way).

Differences from the original, all deliberate:
  * keypoints use the jkp convention, (0, 0) = centre of the top-left pixel of the original image:
    (i + 0.5) * r - 0.5 where the original returns i * r (r = original size / resized size);
  * selection is deterministic: best score first, ties in raster order (torch.argsort leaves ties open).
"""
import math

import torch
import torch.nn.functional as F

import jkpfile

DEFAULT_TOP_K = 4096


def cubic_coeffs(t, a=-0.75):
    """PyTorch's cubic convolution weights for taps at -1, 0, 1, 2 around the floor."""
    def near(x):  # |x| <= 1
        return ((a + 2) * x - (a + 3)) * x * x + 1

    def far(x):  # 1 < |x| < 2
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return [far(t + 1), near(t), near(1 - t), far(2 - t)]


class XFeatFolded:
    def __init__(self, weights_path, device="cpu", top_k=DEFAULT_TOP_K):
        self.meta, tensors = jkpfile.read(weights_path)
        self.w = {k: torch.from_numpy(v.copy()).float().to(device) for k, v in tensors.items()}
        self.device = device
        self.top_k = top_k if top_k > 0 else int(self.meta["top_k"])
        self.threshold = float(self.meta["detection_threshold"])
        self.nms = int(self.meta["nms"])

    # ---------------------------------------------------------------- dense network
    def conv(self, x, name, stride=1, relu=True):
        weight = self.w[name + ".weight"]
        y = F.conv2d(x, weight, self.w[name + ".bias"], stride=stride, padding=weight.shape[-1] // 2)
        return F.relu(y) if relu else y

    def dense(self, image):
        """image: 1x3xHxW float in [0, 1]. Returns (feats, heatmap, logits, stages) on the resized image."""
        h, w = image.shape[2:]
        H, W = h // 32 * 32, w // 32 * 32
        if H == 0 or W == 0:
            raise ValueError("image smaller than 32 pixels")
        st = {"image": image}
        # Resize (bilinear, align_corners = False; the identity when the size is already a multiple of 32),
        # then the mean of the channels and instance norm (biased variance, eps 1e-5).
        x = F.interpolate(image, (H, W), mode="bilinear", align_corners=False)
        st["resized"] = x
        g = x.mean(dim=1, keepdim=True)
        st["gray"] = g
        mean, var = g.mean(), g.var(unbiased=False)
        xn = (g - mean) / torch.sqrt(var + 1e-5)
        st["norm"] = xn

        c = self.conv
        x1 = c(c(c(c(xn, "block1.0"), "block1.1", 2), "block1.2"), "block1.3", 2)
        skip = c(F.avg_pool2d(xn, 4), "skip", relu=False)
        x2 = c(c(x1 + skip, "block2.0"), "block2.1")
        x3 = c(c(c(x2, "block3.0", 2), "block3.1"), "block3.2")
        x4 = c(c(c(x3, "block4.0", 2), "block4.1"), "block4.2")
        x5 = c(c(c(c(x4, "block5.0", 2), "block5.1"), "block5.2"), "block5.3")
        size = x3.shape[-2:]
        x4u = F.interpolate(x4, size, mode="bilinear", align_corners=False)
        x5u = F.interpolate(x5, size, mode="bilinear", align_corners=False)
        feats = c(c(c(x3 + x4u + x5u, "fusion.0"), "fusion.1"), "fusion.2", relu=False)
        heatmap = torch.sigmoid(c(c(c(feats, "heatmap.0"), "heatmap.1"), "heatmap.2", relu=False))
        # Space to depth by 8: channel ky * 8 + kx holds pixel (8y + ky, 8x + kx).
        unfolded = F.pixel_unshuffle(xn, 8)
        logits = c(c(c(c(unfolded, "keypoint.0"), "keypoint.1"), "keypoint.2"), "keypoint.3", relu=False)
        st.update(x1=x1, skip=skip, x2=x2, x3=x3, x4=x4, x5=x5, x4u=x4u, x5u=x5u, feats=feats, heatmap=heatmap,
                  unfolded=unfolded, logits=logits)
        return feats, heatmap, logits, st

    # ---------------------------------------------------------------- sparse sampling
    @staticmethod
    def norm_grid(p, size):
        """float32, in the original's order: 2 * (p / (size - 1)) - 1."""
        return 2.0 * (p.float() / torch.tensor(float(size - 1))) - 1.0

    @staticmethod
    def unnorm(g, size):
        return ((g + 1) * size - 1) / 2

    def sample_nearest(self, m, px, py, H, W):
        """m: HxW. Round half to even; outside = 0."""
        ix = torch.round(self.unnorm(self.norm_grid(px, W), m.shape[1])).long()
        iy = torch.round(self.unnorm(self.norm_grid(py, H), m.shape[0])).long()
        ok = (ix >= 0) & (ix < m.shape[1]) & (iy >= 0) & (iy < m.shape[0])
        out = torch.zeros(len(px), device=m.device)
        out[ok] = m[iy[ok], ix[ok]]
        return out

    def taps(self, m, ix, iy):
        """m: CxHxW, integer positions -> C x N, zero outside."""
        ok = (ix >= 0) & (ix < m.shape[2]) & (iy >= 0) & (iy < m.shape[1])
        v = m[:, iy.clamp(0, m.shape[1] - 1), ix.clamp(0, m.shape[2] - 1)]
        return v * ok.float()

    def sample_bilinear(self, m, px, py, H, W):
        x = self.unnorm(self.norm_grid(px, W), m.shape[2])
        y = self.unnorm(self.norm_grid(py, H), m.shape[1])
        x0, y0 = torch.floor(x), torch.floor(y)
        tx, ty = x - x0, y - y0
        x0, y0 = x0.long(), y0.long()
        return ((1 - tx) * (1 - ty) * self.taps(m, x0, y0) + tx * (1 - ty) * self.taps(m, x0 + 1, y0) +
                (1 - tx) * ty * self.taps(m, x0, y0 + 1) + tx * ty * self.taps(m, x0 + 1, y0 + 1))

    def sample_bicubic(self, m, px, py, H, W):
        x = self.unnorm(self.norm_grid(px, W), m.shape[2])
        y = self.unnorm(self.norm_grid(py, H), m.shape[1])
        x0, y0 = torch.floor(x), torch.floor(y)
        cx, cy = cubic_coeffs(x - x0), cubic_coeffs(y - y0)
        x0, y0 = x0.long(), y0.long()
        out = 0
        for i in range(4):
            row = sum(cx[j] * self.taps(m, x0 - 1 + j, y0 - 1 + i) for j in range(4))
            out = out + cy[i] * row
        return out

    # ---------------------------------------------------------------- detection and description
    def detect(self, feats, heatmap, logits):
        """Returns (pixel positions in the resized image as (N,2) long, scores, descriptors, heat map)."""
        _, _, h8, w8 = logits.shape
        H, W = h8 * 8, w8 * 8
        # Softmax over the 65 channels, the 64 cell channels back to pixels (depth to space).
        heat = F.pixel_shuffle(F.softmax(logits, dim=1)[:, :64], 8)[0, 0]
        p = self.nms // 2
        local_max = F.max_pool2d(heat[None, None], self.nms, stride=1, padding=p)[0, 0]
        cand = (heat == local_max) & (heat > self.threshold)
        iy, ix = torch.nonzero(cand, as_tuple=True)  # raster order
        score = self.sample_nearest(heat, ix, iy, H, W) * self.sample_bilinear(heatmap[0], ix, iy, H, W)[0]
        score[(ix == 0) & (iy == 0)] = -1
        # Best first, ties in raster order (a stable sort of the raster-ordered candidates).
        order = torch.sort(-score, stable=True).indices[: self.top_k]
        ix, iy, score = ix[order], iy[order], score[order]
        keep = score > 0
        ix, iy, score = ix[keep], iy[keep], score[keep]
        feats_n = F.normalize(feats, dim=1)[0]
        desc = F.normalize(self.sample_bicubic(feats_n, ix, iy, H, W).t(), dim=-1)
        return torch.stack([ix, iy], 1), score, desc, heat

    def run(self, rgb, stages=False):
        """rgb: HxWx3 uint8. Returns the same dict layout as xfeat_upstream.run."""
        image = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        h, w = image.shape[2:]
        feats, heatmap, logits, st = self.dense(image)
        pix, score, desc, heat = self.detect(feats, heatmap, logits)
        H, W = logits.shape[2] * 8, logits.shape[3] * 8
        r = torch.tensor([w / W, h / H], dtype=torch.float64, device=pix.device)
        xy = ((pix.double() + 0.5) * r - 0.5).float()
        out = {"keypoints": xy.cpu().numpy(), "scores": score.cpu().numpy(), "descriptors": desc.cpu().numpy()}
        if stages:
            st["kpts_heat"] = heat[None, None]
            out.update({k: v[0].contiguous().cpu().numpy() for k, v in st.items()})
        return out


assert math.isclose(sum(cubic_coeffs(0.3)), 1.0)
