from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
from vesuvius.models.preprocessing.patches.cache import build_cache_params, cache_filename
from vesuvius.models.preprocessing.patches.generate import generate_patch_caches


def _write_pyramid(path: Path, level0: np.ndarray) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.create_array("0", data=level0, chunks=(16, 16, 16))
    root.create_array("1", data=np.ascontiguousarray(level0[::2, ::2, ::2]), chunks=(16, 16, 16))


def _write_config(tmp_path: Path, level: int, find_level: int) -> Path:
    config = {
        "tr_setup": {"model_name": "ome-zarr-resolution-test"},
        "tr_config": {"patch_size": [16, 16, 16], "batch_size": 1},
        "dataset_config": {
            "data_path": str(tmp_path),
            "ome_zarr_resolution": level,
            "valid_patch_find_resolution": find_level,
            "min_labeled_ratio": 0.5,
            "min_bbox_percent": 0.5,
            "targets": {"ink": {"activation": "none"}},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_config_manager_loads_ome_zarr_resolution(tmp_path: Path) -> None:
    _write_pyramid(tmp_path / "images" / "vol.zarr", np.full((64, 64, 64), 100, dtype=np.uint8))
    _write_pyramid(tmp_path / "labels" / "vol_ink.zarr", np.ones((64, 64, 64), dtype=np.uint8))

    mgr = ConfigManager(verbose=False)
    mgr.load_config(_write_config(tmp_path, level=1, find_level=1))
    assert mgr.ome_zarr_resolution == 1
    with pytest.raises(ValueError, match="valid_patch_find_resolution"):
        ConfigManager(verbose=False).load_config(_write_config(tmp_path, level=2, find_level=1))

    # (3) the training level is part of the cache key; level 0 keeps its old key.
    common = dict(data_path=tmp_path, volume_ids=["vol"], patch_size=[16, 16, 16],
                  min_labeled_ratio=0.5, bbox_threshold=0.5, valid_patch_find_resolution=1)
    assert "ome_zarr_resolution" not in build_cache_params(**common).to_dict()
    assert cache_filename(build_cache_params(**common)) != cache_filename(
        build_cache_params(ome_zarr_resolution=1, **common))

    # (2) cached level-0 positions are scaled to level 1 and stay inside the volume.
    generate_patch_caches(_write_config(tmp_path, level=1, find_level=1))
    dataset = ZarrDataset(mgr, is_training=False)
    assert dataset._volumes[0].spatial_shape == (32, 32, 32)
    assert len(dataset._patches) == 8
    assert all(max(p.position) + 16 <= 32 for p in dataset._patches)
