import numpy as np

from vesuvius.neural_tracing.heatmap_single_point.tifxyz import save_tifxyz
from vesuvius.neural_tracing.inference import infer_streamline as m


def test_level_s_crops_land_on_level0_surface(tmp_path, monkeypatch):
    # Level-0 volume with a bright sheet at z=40; level 1 is its 2x downsample.
    level0 = np.zeros((64, 64, 64), dtype=np.uint8)
    level0[40] = 255
    level1 = level0[::2, ::2, ::2]
    yy, xx = np.meshgrid(np.arange(16, 49, 4), np.arange(16, 49, 4), indexing="ij")
    zyxs = np.stack([np.full_like(yy, 40), yy, xx], axis=-1).astype(np.float32)
    save_tifxyz(zyxs, str(tmp_path), "seg", step_size=4, voxel_size_um=8.0, source="test")

    monkeypatch.setattr(m, "VOLUME_SCALE", 1)
    monkeypatch.setattr(m, "DEVICE", "cpu")
    crop_size = (16, 16, 16)
    stored_zyxs, valid = m._load_surface_zyx(tmp_path / "seg")
    cond_direction, _ = m._get_growth_context("left")
    bboxes, _ = m.get_cond_edge_bboxes(stored_zyxs, cond_direction, crop_size, cond_valid=valid)
    assert bboxes

    items = [{"bbox": bbox, "voxels": np.zeros(crop_size, dtype=np.uint8)} for bbox in bboxes]
    ct = m._prepare_rowcol_inputs(level1, items, crop_size, cond_direction)[:, 0].numpy()
    for bbox, crop in zip(bboxes, ct):
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        # The crop lies inside the level-1 array and contains the sheet at z=20.
        assert 0 <= z_min <= 20 <= z_max < level1.shape[0]
        assert 0 <= y_min and y_max < level1.shape[1] and 0 <= x_min and x_max < level1.shape[2]
        assert crop[20 - z_min].min() > 0 > crop.min()
