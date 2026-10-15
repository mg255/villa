"""The resize task must sample one global grid, not restart scale and alignment in every chunk."""
import itertools

import cv2
import numpy as np
from numcodecs import Blosc

from vesuvius.image_proc.run.zarr_tasks.tasks.resize import _resize_chunk_3d, _resize_chunk_worker
from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset


def test_linear_upsample_matches_whole_volume_resize(tmp_path):
    in_shape, out_shape, chunks = (13, 15, 45), (25, 29, 89), (16, 16, 16)
    z, y, x = np.meshgrid(*(np.arange(n) for n in in_shape), indexing="ij")
    ramp = (4 * x + 2 * y + 2 * z + 10).astype(np.uint8)  # 10..238

    level = dict(level_name="0", dtype=np.uint8, compressor=Blosc(cname="zstd", clevel=1))
    src = create_level_dataset(root_group_path=str(tmp_path / "in.zarr"), shape=in_shape, chunks=chunks, **level)
    src[:] = ramp
    out = create_level_dataset(root_group_path=str(tmp_path / "out.zarr"), shape=out_shape, chunks=chunks, **level)

    for start in itertools.product(*(range(0, n, c) for n, c in zip(out_shape, chunks))):
        _resize_chunk_worker(
            (str(tmp_path / "in.zarr"), str(tmp_path / "out.zarr"), "0", *start, chunks,
             in_shape, out_shape, cv2.INTER_LINEAR, None)
        )

    whole = _resize_chunk_3d(ramp, out_shape, cv2.INTER_LINEAR)
    diff = np.abs(out[:].astype(int) - whole.astype(int))
    assert diff.max() <= 1, f"max |chunked - whole| = {diff.max()}"
