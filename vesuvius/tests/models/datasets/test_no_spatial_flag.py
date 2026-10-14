from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from vesuvius.models.datasets import ZarrDataset


def _transform_names(compose):
    return {type(getattr(t, "transform", t)).__name__ for t in compose.transforms}


@pytest.mark.parametrize("no_spatial", [True, False])
def test_zarr_dataset_honours_mgr_no_spatial(tmp_path, no_spatial):
    # --no-spatial and dataset_config.no_spatial both set mgr.no_spatial.
    for sub, name in (("images", "vol.zarr"), ("labels", "vol_surface.zarr")):
        arr = zarr.open(str(tmp_path / sub / name), mode="w", shape=(16, 16, 16), dtype="uint8")
        arr[:] = np.ones((16, 16, 16), dtype=np.uint8)
    mgr = SimpleNamespace(
        data_path=tmp_path,
        train_patch_size=(16, 16, 16),
        targets={"surface": {}},
        skip_patch_validation=True,
        cache_valid_patches=False,
        no_spatial=no_spatial,
    )

    names = _transform_names(ZarrDataset(mgr, is_training=True).transforms)

    spatial = {"Rot90Transform", "TransposeAxesTransform"}
    if no_spatial:
        assert not names & spatial
    else:
        assert spatial <= names
