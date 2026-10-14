"""Voxels no patch reached are background when the blended store carries coverage.

blend_logits writes a uint8 ``coverage`` array inside its output store (1 where
any patch weight landed). finalize_outputs writes 0 where coverage is 0, in
every mode and for single-channel models; a store without it is finalized as
before. Voxel 2 is covered but has logits exactly 0: it keeps p = 0.5.
"""

import os

import numpy as np
import pytest

from vesuvius.data.utils import open_zarr
from vesuvius.models.run.finalize_outputs import finalize_logits

# (case, channels, mode, threshold, expected with coverage, expected without coverage)
CASES = [
    ("threshold", 2, "binary", 0.2, [[0, 0, 255, 255]], [[255, 255, 255, 255]]),
    ("probability", 2, "binary", None, [[0, 0, 127, 255]], [[127, 127, 127, 255]]),
    ("multiclass", 2, "multiclass", None,
     [[0, 0, 128, 0], [0, 0, 128, 255], [0, 0, 0, 1]],
     [[128, 128, 128, 0], [128, 128, 128, 255], [0, 0, 0, 1]]),
    ("multiclass_argmax", 2, "multiclass", 0.5, [[0, 0, 0, 1]], [[0, 0, 0, 1]]),
    ("sigmoid", 1, "binary", None, [[0, 0, 127, 255]], [[127, 127, 127, 255]]),
    ("sigmoid_threshold", 1, "binary", 0.2, [[0, 0, 255, 255]], [[255, 255, 255, 255]]),
]


def _blended_store(path, channels, with_coverage):
    """(C, 1, 1, 4) logits: voxels 0-1 uncovered (0), 2 covered at 0, 3 covered foreground."""
    logits = np.zeros((channels, 1, 1, 4), dtype=np.float16)
    logits[-1, 0, 0, 3] = 20.0
    store = open_zarr(path, mode="w", shape=logits.shape, chunks=(1, 1, 1, 4), dtype=np.float16, fill_value=0)
    store[:] = logits
    if with_coverage:
        cov = open_zarr(os.path.join(path, "coverage"), mode="w", shape=(1, 1, 4), chunks=(1, 1, 4),
                        dtype=np.uint8, fill_value=0)
        cov[:] = np.array([0, 0, 1, 1], dtype=np.uint8).reshape(1, 1, 4)


@pytest.mark.parametrize("with_coverage", [True, False], ids=["coverage", "no_coverage"])
@pytest.mark.parametrize("case,channels,mode,threshold,expected_cov,expected_old", CASES,
                         ids=[c[0] for c in CASES])
def test_finalize_masks_uncovered_voxels(tmp_path, with_coverage, case, channels, mode, threshold,
                                         expected_cov, expected_old):
    src, dst = str(tmp_path / "blended.zarr"), str(tmp_path / "final.zarr")
    _blended_store(src, channels, with_coverage)
    finalize_logits(src, dst, mode=mode, threshold=threshold, num_workers=1, verbose=False)
    out = open_zarr(dst, mode="r")[:, 0, 0]
    assert out.tolist() == (expected_cov if with_coverage else expected_old)
