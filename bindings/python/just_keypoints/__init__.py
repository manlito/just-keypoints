"""just-keypoints: ALIKED and RaCo keypoints with minimal dependencies.

    import numpy as np, just_keypoints as jkp
    extractor = jkp.Extractor(top_k=1000)                  # CPU, aliked-n16, built-in weights
    features = extractor.extract(image)                    # (h, w), (h, w, 3) or (h, w, 4) uint8
    features.xy, features.score, features.desc             # (n, 2), (n,), (n, 128) float32

The compiled core is in ``just_keypoints._core``; everything in it is re-exported here. Pixel coordinates are
x right, y down, with (0, 0) the centre of the top-left pixel.
"""

from __future__ import annotations

import numpy as np

from ._build_info import CUDA_VERSION, VARIANT, __version__
from . import _core
from ._core import Features, backend_available, compiled_backends

__all__ = ["CUDA_VERSION", "VARIANT", "__version__", "Extractor", "Features", "backend_available",
           "compiled_backends"]


class Extractor(_core.Extractor):
    __doc__ = _core.Extractor.__doc__

    def extract(self, image) -> Features:
        """Keypoints of one image: a uint8 array of shape (h, w) (gray), (h, w, 3) (RGB) or (h, w, 4)
        (RGBA), rows from the top. Other memory layouts (a view, a crop) are copied first."""
        a = np.asarray(image)
        if a.dtype != np.uint8:
            raise TypeError(f"image must be uint8, not {a.dtype} (scale floats to 0..255 first)")
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[:, :, 0]
        return self._extract(np.ascontiguousarray(a))

    def __repr__(self) -> str:
        return f"Extractor(backend={self.backend!r}, model={self.model!r}, precision={self.precision!r})"
