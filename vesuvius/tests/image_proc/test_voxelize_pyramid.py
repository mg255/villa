"""voxelize_objs pyramid levels must keep a 1-voxel-thick surface: every block that holds surface stays set."""
import sys

import numpy as np
import pytest

pytest.importorskip("trimesh")
pytest.importorskip("numba")
import trimesh  # noqa: E402
import zarr  # noqa: E402

from vesuvius.image_proc.run import voxelize_objs as vox  # noqa: E402

N = 32


def _any_pool(a, f):
    return a.reshape(N // f, f, N // f, f, N // f, f).max(axis=(1, 3, 5))


def test_label_pyramid_keeps_thin_surface(tmp_path, monkeypatch):
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    gx, gy = np.meshgrid(np.linspace(1, 30, 30), np.linspace(1, 30, 30))
    verts = np.stack([gx, gy, 6 + 0.45 * gx + 0.3 * gy], axis=-1).reshape(-1, 3)
    i = np.arange(29 * 30).reshape(29, 30)[:, :29].ravel()
    faces = np.concatenate([np.stack([i, i + 1, i + 30], 1), np.stack([i + 1, i + 31, i + 30], 1)])
    trimesh.Trimesh(verts, faces, process=False).export(mesh_dir / "plane.obj")

    out = tmp_path / "out.zarr"
    monkeypatch.setattr(sys, "argv", ["voxelize_objs", str(mesh_dir), "--spatial-shape", str(N), str(N), str(N),
                                      "--output_path", str(out), "--num_workers", "1", "--chunk_size", "8"])
    vox.main()
    root = zarr.open_group(str(out), mode="r")
    level0 = root["0"][:]

    # Level 0 is the plain slice rasterisation, untouched by the pyramid code.
    v, t, lab = vox.process_mesh(str(mesh_dir / "plane.obj"), 0, False, False, None, np.dtype("uint16"))[:3]
    expected0 = np.stack([vox.process_slice_points_label(v, t, lab, z, N, N) for z in range(N)])
    np.testing.assert_array_equal(level0, expected0)
    assert level0.any()

    for level in (1, 2):
        np.testing.assert_array_equal(root[str(level)][:], _any_pool(level0, 2 ** level))
