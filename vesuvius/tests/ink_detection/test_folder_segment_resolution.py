from pathlib import Path
from types import SimpleNamespace

import torch
import zarr

from vesuvius.ink_detection.inference.infer import (
    infer_folder,
    resolve_segment_zarr_path,
)


def _pyramid(path: Path) -> Path:
    group = zarr.open_group(path, mode="w", zarr_format=2)
    for level in ("0", "1"):
        group.create_array(level, shape=(3, 4, 4), dtype="u1")
    return path


def _run_folder(folder: Path, monkeypatch) -> tuple[int, list[Path]]:
    ran = []
    monkeypatch.setattr(
        "vesuvius.ink_detection.inference.infer.infer_single_zarr",
        lambda **kwargs: ran.append(kwargs["input_zarr"]),
    )
    checkpoint = folder.parent / "model.pth"
    checkpoint.write_bytes(b"weights")
    args = SimpleNamespace(
        folder=folder, checkpoint=checkpoint, output_prefix="",
        direction="forward", resolution="0", mask_path=None, overlap=0.5,
        stride=None, blend_mode="auto", layer_start=None, layer_end=None,
        tta_mirror=False, tta_batch_size=None,
    )
    configured = SimpleNamespace(patch_size=16, amp_dtype=None)
    return infer_folder(args, configured, device=torch.device("cpu")), ran


def test_ink_9um_layout_resolves_surface_volume_and_skips_labels(
    tmp_path, monkeypatch
):
    segment = tmp_path / "family" / "w035"
    volume = _pyramid(segment / "surface-volume.zarr")
    _pyramid(segment / "w035_inklabels.zarr")
    _pyramid(segment / "w035_supervision_mask.zarr")

    assert resolve_segment_zarr_path(segment) == volume
    assert _run_folder(segment.parent, monkeypatch) == (0, [volume])
    assert _run_folder(segment, monkeypatch) == (0, [volume])

    tutorial = tmp_path / "tutorial" / "seg"
    tutorial_volume = _pyramid(tutorial / "seg.zarr")
    _pyramid(tutorial / "seg_inklabels.zarr")
    assert resolve_segment_zarr_path(tutorial) == tutorial_volume
    assert _run_folder(tutorial.parent, monkeypatch) == (0, [tutorial_volume])

    labels_only = tmp_path / "labels_only" / "w035"
    _pyramid(labels_only / "w035_inklabels.zarr")
    (labels_only / "notes").mkdir()
    assert _run_folder(labels_only.parent, monkeypatch) == (1, [])
