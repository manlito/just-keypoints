"""Tests of the Python package: the CPU backend with the built-in weights, and each GPU backend the build has
against the CPU backend when this machine can run it (skipped otherwise, as on CI's runners)."""

from pathlib import Path

import numpy as np
import pytest

import just_keypoints as jkp

WEIGHTS = Path(__file__).resolve().parents[3] / "weights"


def picture(w=320, h=240, seed=1):
    """A deterministic RGB test picture: gradient, rectangles, discs and a little noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([60 + 80 * x / w, 60 + 80 * y / w, 60 + 80 * y / h], axis=-1)
    for _ in range(40):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rx, ry = 3 + rng.integers(0, w // 5), 3 + rng.integers(0, h // 5)
        mask = (np.abs(x - cx) < rx) & (np.abs(y - cy) < ry)
        if rng.integers(0, 3) == 0:
            mask &= ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
        img[mask] = rng.integers(0, 256, 3)
    img += rng.integers(-4, 5, (h, w, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def image():
    return picture()


@pytest.fixture(scope="module")
def aliked():
    return jkp.Extractor(top_k=500)


def test_build_info():
    assert jkp.__version__.count(".") == 2
    assert jkp.VARIANT in ("cpu", "cu12", "cu13", "source")
    assert "cpu" in jkp.compiled_backends()
    assert jkp.backend_available("cpu")
    assert (jkp.CUDA_VERSION != "") == jkp.backend_available("cuda")
    if jkp.VARIANT.startswith("cu"):
        assert jkp.backend_available("cuda") and jkp.CUDA_VERSION.startswith(jkp.VARIANT[2:])
    with pytest.raises(ValueError):
        jkp.backend_available("tpu")


def test_aliked(aliked, image):
    f = aliked.extract(image)
    assert f.n == len(f) == 500 and f.dim == 128
    assert f.xy.shape == (500, 2) and f.xy.dtype == np.float32
    assert f.score.shape == (500,)
    assert f.desc.shape == (500, 128)
    assert f.rank is None and f.cov is None
    np.testing.assert_allclose(np.linalg.norm(f.desc, axis=1), 1, atol=1e-4)
    assert (f.xy >= -0.5).all() and (f.xy[:, 0] <= image.shape[1]).all() and (f.xy[:, 1] <= image.shape[0]).all()
    assert ((f.score >= 0) & (f.score <= 1)).all()
    g = aliked.extract(image)  # same input, same output; and the first result is a copy that stays valid
    np.testing.assert_array_equal(f.xy, g.xy)
    np.testing.assert_array_equal(f.desc, g.desc)


def test_threshold(image):
    loose = jkp.Extractor(score_threshold=0.2).extract(image)
    strict = jkp.Extractor(score_threshold=0.5).extract(image)
    assert 0 < strict.n < loose.n
    assert jkp.Extractor(score_threshold=0.2, max_keypoints=50).extract(image).n == min(50, loose.n)


def test_image_layouts(aliked, image):
    ref = aliked.extract(image)
    rgba = np.concatenate([image, np.full(image.shape[:2] + (1,), 7, np.uint8)], axis=2)
    np.testing.assert_array_equal(aliked.extract(rgba).xy, ref.xy)  # alpha is ignored
    padded = np.zeros((image.shape[0], image.shape[1] + 8, 3), np.uint8)
    padded[:, :-8] = image
    np.testing.assert_array_equal(aliked.extract(padded[:, :-8]).xy, ref.xy)  # a view is copied
    gray = aliked.extract(image.mean(axis=2).astype(np.uint8))
    assert gray.n == 500
    np.testing.assert_array_equal(aliked.extract(image.mean(axis=2).astype(np.uint8)[:, :, None]).xy, gray.xy)
    with pytest.raises(TypeError):
        aliked.extract(image.astype(np.float32))
    with pytest.raises(ValueError):
        aliked.extract(image[:, :, :2])


def test_raco(image):
    f = jkp.Extractor(model="raco", top_k=300).extract(image)
    assert f.n == 300 and f.dim == 0 and f.desc is None
    assert f.rank.shape == (300,) and f.cov.shape == (300, 2, 2)
    np.testing.assert_allclose(f.cov[:, 0, 1], f.cov[:, 1, 0], rtol=1e-5, atol=1e-6)
    d = jkp.Extractor(model="raco", top_k=300, descriptor="aliked-n16", skip_covariance=True).extract(image)
    assert d.dim == 128 and d.desc.shape == (300, 128) and d.cov is None and d.rank is not None
    np.testing.assert_array_equal(d.xy, f.xy)


def test_errors():
    with pytest.raises(ValueError):
        jkp.Extractor(backend="tpu")
    with pytest.raises(ValueError):
        jkp.Extractor(precision="fp8")
    with pytest.raises(RuntimeError, match="fp32"):
        jkp.Extractor(precision="fp16")  # the CPU backend runs in FP32 only
    with pytest.raises(RuntimeError):
        jkp.Extractor(weights=b"not a weights file")
    with pytest.raises(RuntimeError):
        jkp.Extractor(model="aliked-n32")  # not built in
    with pytest.raises(TypeError):
        jkp.Extractor(weights=3)
    for b in ("cuda", "metal", "webgpu"):
        if not jkp.backend_available(b):
            with pytest.raises(RuntimeError, match="not available"):
                jkp.Extractor(backend=b)


@pytest.mark.skipif(not (WEIGHTS / "aliked-n16.jkpw").exists(), reason="needs the repository's weights/")
def test_weights_from_file_and_bytes(aliked, image):
    ref = aliked.extract(image)
    path = WEIGHTS / "aliked-n16.jkpw"
    for w in (path, str(path), path.read_bytes()):
        np.testing.assert_array_equal(jkp.Extractor(top_k=500, weights=w).extract(image).desc, ref.desc)


def gpu(backend, **kw):
    if not jkp.backend_available(backend):
        pytest.skip(f"{backend} is not in this build")
    try:
        return jkp.Extractor(backend=backend, **kw)
    except RuntimeError as e:  # no device on this machine
        pytest.skip(f"{backend}: {e}")


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_gpu_matches_cpu(backend, image):
    cpu = jkp.Extractor(top_k=500).extract(image)
    f = gpu(backend, top_k=500).extract(image)
    assert f.n == cpu.n
    np.testing.assert_allclose(f.xy, cpu.xy, atol=1e-2)
    np.testing.assert_allclose(f.desc, cpu.desc, atol=1e-3)
    h = gpu(backend, top_k=500, precision="fp16").extract(image)
    assert h.n == 500
    r_cpu = jkp.Extractor(model="raco", top_k=300).extract(image)
    r = gpu(backend, model="raco", top_k=300).extract(image)
    np.testing.assert_allclose(r.xy, r_cpu.xy, atol=1e-2)
    np.testing.assert_allclose(r.cov, r_cpu.cov, rtol=1e-3, atol=1e-3)
