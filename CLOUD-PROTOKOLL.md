# CLOUD-PROTOKOLL: infer_streamline level-s crop coordinates

Session: https://claude.ai/code/session_01K2riy556CqV7ibkYzsx3L5 (cloud container, 2026-10-15).
Fix branch: `streamline-level-coordinates` @ `d6e37cb16ce993cf1906b293a0c19936bc37393f`
(on top of `upstream/main` = ScrollPrize/villa `5432721dcee855b7881a6da646ac9a36fb2ae9a8`, "Photogrammetry, again (#2103)").
This protocol branch: `streamline-level-coordinates-protokoll` = fix commit + this file.
No PR, no comments, nothing pushed to ScrollPrize/villa.

## 0. Prior-work search (ScrollPrize/villa, open + closed, via GitHub MCP search)

| query | PRs | issues |
|---|---|---|
| `infer_streamline` | 0 | 0 |
| `streamline` | #1796, #1668, #2077, #960 ("Streamline trainer + new copy model inference ..." – the original code drop, closed 2026-05), #1104, #282 – all closed, none about crop coordinates / volume scale | 1 hit: #377 (VC3D coordinates vs. max scale, 2025, closed, unrelated tool) |
| `volume_scale` | #1284 (opt-in displacement scaling across voxel sizes, closed), #1821, #990, #773 – none touch infer_streamline crops | #1319, #1402, #377, #497 – tifxyz/VC3D scale topics, not infer_streamline |
| `retarget` | #1284, #1778, #1246 – closed, unrelated | #1673, #1694, #639 – unrelated |
| `streamline is:open` | 0 | – |

Conclusion: nobody is working on this; proceeded.

## 1. Branch

```
git remote add upstream https://github.com/ScrollPrize/villa.git
git fetch upstream main
git checkout -b streamline-level-coordinates upstream/main   # 5432721dcee855b7881a6da646ac9a36fb2ae9a8
```

## 2. Fix

How the rest of the code treats coordinates decided the approach. The training dataset retargets
every segment to level s (`datasets/dataset_rowcol_cond.py:118`, `seg.retarget(2 ** volume_scale)`), and so do the sister
inference scripts (`inference/common.py:1201`, `infer_rowcol_triplet_wraps.py:2626`). infer_streamline builds bboxes,
voxelized surface masks, CT crops, the merged output window and the streamline integration in **one** frame where one crop
voxel equals one world voxel. So the surface must be retargeted into level s before bboxes/crops are built (taking crops at
level 0 would feed the model a resolution it was not trained on). Two level-0 quantities have to follow:
the tifxyz step (`TIFXYZ_VOXEL_STEP`, derived from meta.json `1/scale`, i.e. level-0 voxels) is divided by `2**s` where it
drives integration distance and front spacing, and the merged output is multiplied back by `2**s` before saving, so the
written tifxyz is in level-0 voxels like its input (which is what the next iteration reads).

8 insertions, 2 deletions. At s = 0 every changed expression is an identity (`/ 2**0`, `* 2**0`, retarget skipped), and the
s = 0 runs below are byte-identical between main and branch.

```diff
diff --git a/vesuvius/src/vesuvius/neural_tracing/inference/infer_streamline.py b/vesuvius/src/vesuvius/neural_tracing/inference/infer_streamline.py
index af4e162..c0c32b8 100644
--- a/vesuvius/src/vesuvius/neural_tracing/inference/infer_streamline.py
+++ b/vesuvius/src/vesuvius/neural_tracing/inference/infer_streamline.py
@@ -617,6 +617,8 @@ def _load_surface_zyx(tifxyz_path):
         raise NotADirectoryError(f"tifxyz path must be a directory: {tifxyz_path}")
 
     surface = read_tifxyz(tifxyz_path)
+    if int(VOLUME_SCALE):  # tifxyz is in level-0 voxels; crops come from level VOLUME_SCALE
+        surface = surface.retarget(2 ** int(VOLUME_SCALE))
     surface.use_stored_resolution()
     x, y, z, valid = surface[:]
     stored_zyxs = np.stack([z, y, x], axis=-1)
@@ -1350,7 +1352,7 @@ def _require_avg_field(avg_group, name, channels):
 
 
 def _integration_step_sizes():
-    tifxyz_step = float(TIFXYZ_VOXEL_STEP)
+    tifxyz_step = float(TIFXYZ_VOXEL_STEP) / 2 ** int(VOLUME_SCALE)
     tifxyz_steps = int(TIFXYZ_STEPS)
     step_size = float(INTEGRATION_STEP_SIZE)
     if tifxyz_step <= 0.0:
@@ -2035,7 +2037,7 @@ def integrate_streamlines_from_edge(avg_group, edge_zyx, edge_axis_scale, window
     integration.create_dataset("diagnostic_validity_oob", data=diag_validity_oob, chunks=(min(n_steps, 4096),))
     integration.create_dataset("diagnostic_validity_nonfinite", data=diag_validity_nonfinite, chunks=(min(n_steps, 4096),))
     integration.create_dataset("diagnostic_low_validity", data=diag_low_validity, chunks=(min(n_steps, 4096),))
-    integration.attrs["tifxyz_voxel_step"] = float(TIFXYZ_VOXEL_STEP)
+    integration.attrs["tifxyz_voxel_step"] = float(TIFXYZ_VOXEL_STEP) / 2 ** int(VOLUME_SCALE)
     integration.attrs["tifxyz_steps"] = int(TIFXYZ_STEPS)
     integration.attrs["integration_steps"] = int(len(step_sizes))
     integration.attrs["integration_step_size"] = float(INTEGRATION_STEP_SIZE)
@@ -2367,6 +2369,8 @@ def save_merged_streamline_tifxyz(
         cond_direction,
         integration_group,
     )
+    # Back to level-0 voxels so the saved tifxyz matches its input.
+    merged_zyxs = np.where((merged_zyxs == -1).all(axis=-1, keepdims=True), -1.0, merged_zyxs * 2 ** int(VOLUME_SCALE))
     output_dir = _output_tifxyz_dir() if output_dir is None else Path(output_dir)
     output_uuid = _output_tifxyz_uuid(
         iteration_idx=iteration_idx,
@@ -2582,6 +2586,8 @@ def _run_streamline_iteration(
         integration_margin=bbox_integration_margin,
     )
     surface = read_tifxyz(input_tifxyz_path)
+    if int(VOLUME_SCALE):
+        surface = surface.retarget(2 ** int(VOLUME_SCALE))
     edge_axis_scale = surface._scale[0] if _get_direction_spec(cond_direction)["axis"] == "col" else surface._scale[1]
     output_path = _output_zarr_path(iteration_idx)
     expected_output_tifxyz_path = None
```

## 3. Regression test

`vesuvius/tests/neural_tracing/test_infer_streamline_level_coordinates.py` (31 lines): 64³ level-0 array with a bright sheet
at z = 40, level 1 = `[::2, ::2, ::2]`, a flat tifxyz on that sheet in level-0 voxels (step 4), `VOLUME_SCALE = 1`.
It runs the real `_load_surface_zyx` → `get_cond_edge_bboxes` → `_prepare_rowcol_inputs` and asserts every crop is inside
the level-1 array and contains the sheet at level-1 z = 20.

Same test file run against both checkouts (`PYTHONPATH=<checkout>/vesuvius/src python -m pytest -q -p no:cacheprovider test_infer_streamline_level_coordinates.py`), verbatim:

```
=== main
>           assert 0 <= z_min <= 20 <= z_max < level1.shape[0]
E           assert 32 <= 20
1 failed in 4.82s
=== branch
1 passed in 3.50s
```

## 4. Real data (Scroll 5 / PHerc0172)

Data source: public bucket `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/`. `dl.ash2txt.org` and
`data.aws.ash2txt.org` are blocked by this container's egress policy (CONNECT 403), so the "53keV_7.91um_Scroll5.zarr"
from the ash2txt server could not be used directly. Its S3 counterpart was used instead:

- Volume: `PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr` (OME-zarr v2, uint8, blosc/zstd, 128³ chunks,
  `/` separator). Level 0 shape zyx `[21000, 6700, 9100]`, level 1 `[10500, 3350, 4550]`. The segment directory names
  `...-on-20241024131838-7.91um.tifxyz` refer to this volume id.
- Segment: the smallest of the 53 `*-on-20241024131838-7.91um.tifxyz` meshes in `PHerc0172/segments/`:
  `PHerc0172/segments/20250926145427-w084_20250926145427439_flatboi/mesh/20250926145427-on-20241024131838-7.91um.tifxyz/`

  | file | bytes | sha256 |
  |---|---|---|
  | meta.json | 285 | `e334bdf1c9a298221d91a44d21bc3c72cb236b5cf2a4a7701bf768285b05c679` |
  | x.tif | 597040 | `9e60c244aa0c0538b61311459df7828dcbaf101bf9148ad1b96ce211ce3fa92a` |
  | y.tif | 640502 | `857da844944ce14b7938b220e7f8397f6a70a50779f193b605f790d7b8bbd0e2` |
  | z.tif | 460926 | `87c6f53b743b7d1ecb8b42d9d9ebac7de912bebbd4eaa4a97519b9bb78f6fc18` |
  | total | 1698753 | |

  meta.json scale `[0.05, 0.05]` (tifxyz step 20 level-0 voxels); bbox x 1205.9–7693.9, y 663.0–5928.5, z 2622.4–11415.3.
- Volume bytes downloaded (only the chunks covering the 12 CT crops per run): level 0: 207 chunks, 321,791,605 bytes
  (+1 absent key = masked/empty chunk, fill 0); level 1: 170 chunks, 258,496,305 bytes (+1 absent key). The other two runs
  reused the mirror (main s=1 reads nothing: every crop is outside the array). Mirror on disk: 605,766,080 bytes.

### Method

Harness `crop_evidence.py` (Appendix A) imports `infer_streamline` from the given checkout, sets `VOLUME_SCALE = s`,
`TIFXYZ_VOXEL_STEP = derive_tifxyz_voxel_step(seg)` (as `run_from_args` does), crop size `[96, 256, 256]` from
`configs/config_rowcol_cond_dense.json`, all other module defaults (grow direction `left`, overlap 0.5, ...), and calls the
**real** `_run_streamline_iteration`. `run_rowcol_bbox_inference` is replaced by a stub that captures its arguments
(surface, bboxes, stored coordinates) and stops, so no model is needed. Then:

- crop placement over **all** bboxes against the shape of the opened level-s array;
- for 12 evenly spaced bboxes: real `upsample_voxelize_tifxyz_surface_in_bboxes` (same call as `_run_bbox_batch`) and real
  `_prepare_rowcol_inputs` (CT channel 0 = per-crop z-score from `_read_volume_crop`); raw CT values via
  `_read_volume_crop(..., "unit") * 255`.
- "at points" = CT at the stored tifxyz vertices (in the coordinates the crops were built from) that fall inside the crop;
  "at mask" = CT at voxels of the model's surface-mask channel.

`mask_coverage.py` (Appendix B) counts, over all bboxes, how many crops get a non-empty surface-mask channel.

Commands (venv in the scratchpad, see §6):

```
python -I crop_evidence.py <checkout>/vesuvius/src <seg_w084> <vol_mirror> <s> 12 <out.json>    # s in {0, 1}, checkout in {main, branch}
python -I mask_coverage.py <checkout>/vesuvius/src <seg_w084> <vol_mirror> <s> 0 <out.json>
```

### Results

| run | s | level shape zyx | crops | fully outside | not fully inside | raw CT at surface pts | z-score at surface pts | raw CT crop mean | nonzero frac | crops with mask (all) | 12 CT crops with mask, z-score at mask | sha256(bboxes+inputs) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main_s0 | 0 | [21000, 6700, 9100] | 479 | 0/479 (0 %) | 0/479 (0 %) | 144.9 | +0.107 | 142.7 | 0.992 | 67/479 | 3/12, +0.159 | `84b92c7287b7c683` |
| branch_s0 | 0 | [21000, 6700, 9100] | 479 | 0/479 (0 %) | 0/479 (0 %) | 144.9 | +0.107 | 142.7 | 0.992 | 67/479 | 3/12, +0.159 | `84b92c7287b7c683` |
| main_s1 | 1 | [10500, 3350, 4550] | 479 | 479/479 (100 %) | 479/479 (100 %) | +0.000 | +0.000 | +0.000 | 0.000 | 67/479 | 3/12, +0.000 | `aafad37f6250a80b` |
| branch_s1 | 1 | [10500, 3350, 4550] | 180 | 0/180 (0 %) | 0/180 (0 %) | 145.5 | +0.135 | 138.0 | 0.948 | 3/180 | 0/12, n/a (no mask voxels) | `a699975e7c059917` |

Reading:

- **s = 0:** main and branch are byte-identical (same bboxes and model inputs, sha256 `84b92c72…`).
- **s = 1, main:** all 479 of 479 crops lie fully outside the level-1 array, and the CT channel is 0 everywhere (raw mean 0.0, nonzero
  fraction 0.000). This reproduces the reported behaviour (207/207 there; 479/479 for this segment).
- **s = 1, branch:** 0 of 180 crops are outside. CT at the surface points is a raw mean of 145.5 with z-score +0.135, against 144.9 / +0.107 at s = 0 for
  the same segment. The crop positions are ≈ ½ of the level-0 ones (e.g. z 3627 vs 7262), as expected. Fewer crops
  (180 vs 479) because each level-1 crop spans twice the level-0 extent per axis.
- The z-score at the surface here (+0.11 … +0.14) is much lower than the +0.628 you measured. I can't reconcile the
  two: they come from a different segment and possibly a different statistic, so I do not claim to reproduce that number.

**The effect on predictions is not measured.** There is no public streamline checkpoint, so no model was run, and nothing
here says whether traces get better. The evidence covers only the model *inputs*: where the crops land and what CT they contain.

### Summary JSON per run (verbatim, `per_crop` omitted)

```
{"src": "/tmp/claude-0/-home-user-villa/8a2abe69-a0a7-5034-b355-e37b8410ca16/scratchpad/wt_main/vesuvius/src", "tifxyz": "/tmp/claude-0/-home-user-villa/8a2abe69-a0a7-5034-b355-e37b8410ca16/scratchpad/data/seg_w084", "volume": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr", "volume_scale": 0, "level_shape_zyx": [21000, 6700, 9100], "crop_size": [96, 256, 256], "tifxyz_voxel_step": 20.0, "edge_axis_scale_surface_scale": [0.05, 0.05], "stored_zyx_min": [2622.42333984375, 663.029296875, 1205.9190673828125], "stored_zyx_max": [11415.255859375, 5928.517578125, 7693.9267578125], "n_crops": 479, "crops_fully_inside": 479, "crops_not_fully_inside": 0, "crops_fully_outside": 0, "share_not_fully_inside": 0.0, "share_fully_outside": 0.0, "ct_crop_indices": [0, 43, 87, 130, 174, 217, 261, 304, 348, 391, 435, 478], "ct_crops_with_surface_voxels": 3, "mean_raw_at_surface": 141.0160878499349, "mean_raw_crop": 142.73916498819986, "mean_raw_nonzero_frac": 0.9922973314921061, "mean_zscore_at_surface": 0.15942831337451935, "ct_crops_with_stored_points": 12, "mean_raw_at_points": 144.90802319844565, "mean_zscore_at_points": 0.1072152922473227, "sha256_bboxes_and_inputs": "84b92c7287b7c683f6891da2c00f97a0f0f41545a69d76daf9126d9dfff5e467", "download": {"fetched_chunks": 207, "absent_chunks": 1, "fetched_bytes": 321791605}}
{"src": "/home/user/villa/vesuvius/src", "tifxyz": "/tmp/claude-0/-home-user-villa/8a2abe69-a0a7-5034-b355-e37b8410ca16/scratchpad/data/seg_w084", "volume": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr", "volume_scale": 0, "level_shape_zyx": [21000, 6700, 9100], "crop_size": [96, 256, 256], "tifxyz_voxel_step": 20.0, "edge_axis_scale_surface_scale": [0.05, 0.05], "stored_zyx_min": [2622.42333984375, 663.029296875, 1205.9190673828125], "stored_zyx_max": [11415.255859375, 5928.517578125, 7693.9267578125], "n_crops": 479, "crops_fully_inside": 479, "crops_not_fully_inside": 0, "crops_fully_outside": 0, "share_not_fully_inside": 0.0, "share_fully_outside": 0.0, "ct_crop_indices": [0, 43, 87, 130, 174, 217, 261, 304, 348, 391, 435, 478], "ct_crops_with_surface_voxels": 3, "mean_raw_at_surface": 141.0160878499349, "mean_raw_crop": 142.73916498819986, "mean_raw_nonzero_frac": 0.9922973314921061, "mean_zscore_at_surface": 0.15942831337451935, "ct_crops_with_stored_points": 12, "mean_raw_at_points": 144.90802319844565, "mean_zscore_at_points": 0.1072152922473227, "sha256_bboxes_and_inputs": "84b92c7287b7c683f6891da2c00f97a0f0f41545a69d76daf9126d9dfff5e467", "download": {"fetched_chunks": 0, "absent_chunks": 0, "fetched_bytes": 0}}
{"src": "/tmp/claude-0/-home-user-villa/8a2abe69-a0a7-5034-b355-e37b8410ca16/scratchpad/wt_main/vesuvius/src", "tifxyz": "/tmp/claude-0/-home-user-villa/8a2abe69-a0a7-5034-b355-e37b8410ca16/scratchpad/data/seg_w084", "volume": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr", "volume_scale": 1, "level_shape_zyx": [10500, 3350, 4550], "crop_size": [96, 256, 256], "tifxyz_voxel_step": 20.0, "edge_axis_scale_surface_scale": [0.05, 0.05], "stored_zyx_min": [2622.42333984375, 663.029296875, 1205.9190673828125], "stored_zyx_max": [11415.255859375, 5928.517578125, 7693.9267578125], "n_crops": 479, "crops_fully_inside": 0, "crops_not_fully_inside": 479, "crops_fully_outside": 479, "share_not_fully_inside": 1.0, "share_fully_outside": 1.0, "ct_crop_indices": [0, 43, 87, 130, 174, 217, 261, 304, 348, 391, 435, 478], "ct_crops_with_surface_voxels": 3, "mean_raw_at_surface": 0.0, "mean_raw_crop": 0.0, "mean_raw_nonzero_frac": 0.0, "mean_zscore_at_surface": 0.0, "ct_crops_with_stored_points": 12, "mean_raw_at_points": 0.0, "mean_zscore_at_points": 0.0, "sha256_bboxes_and_inputs": "aafad37f6250a80b46bea9ce04e51923f2a54faec67eb58fb76b1529f1e88f84", "download": {"fetched_chunks": 0, "absent_chunks": 0, "fetched_bytes": 0}}
{"src": "/home/user/villa/vesuvius/src", "tifxyz": "/tmp/claude-0/-home-user-villa/8a2abe69-a0a7-5034-b355-e37b8410ca16/scratchpad/data/seg_w084", "volume": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr", "volume_scale": 1, "level_shape_zyx": [10500, 3350, 4550], "crop_size": [96, 256, 256], "tifxyz_voxel_step": 20.0, "edge_axis_scale_surface_scale": [0.1, 0.1], "stored_zyx_min": [1311.211669921875, 331.5146484375, 602.9595336914062], "stored_zyx_max": [5707.6279296875, 2964.2587890625, 3846.96337890625], "n_crops": 180, "crops_fully_inside": 180, "crops_not_fully_inside": 0, "crops_fully_outside": 0, "share_not_fully_inside": 0.0, "share_fully_outside": 0.0, "ct_crop_indices": [0, 16, 33, 49, 65, 81, 98, 114, 130, 146, 163, 179], "ct_crops_with_surface_voxels": 0, "mean_raw_at_surface": null, "mean_raw_crop": 138.02652740478516, "mean_raw_nonzero_frac": 0.9475390116373698, "mean_zscore_at_surface": null, "ct_crops_with_stored_points": 12, "mean_raw_at_points": 145.46575673421225, "mean_zscore_at_points": 0.1349067895207554, "sha256_bboxes_and_inputs": "a699975e7c059917485b6015d17712da9521e239efba3398553e07a7d91c4a79", "download": {"fetched_chunks": 170, "absent_chunks": 1, "fetched_bytes": 258496305}}
```

### Per-crop values for the 12 CT crops (verbatim from the run JSON)

`main_s0` (crop indices [0, 43, 87, 130, 174, 217, 261, 304, 348, 391, 435, 478]):

| bbox min zyx | mask voxels | stored pts in crop | raw CT at pts | z-score at pts | raw CT crop mean |
|---|---|---|---|---|---|
| [7262, 2486, 7171] | 31113 | 101 | 139.5 | +0.093 | 138.0 |
| [8021, 2562, 7208] | 0 | 80 | 137.7 | +0.006 | 137.6 |
| [8746, 2692, 7115] | 0 | 43 | 136.0 | -0.140 | 138.5 |
| [9581, 2334, 7067] | 50094 | 141 | 140.2 | +0.166 | 137.4 |
| [9951, 5373, 6336] | 0 | 224 | 119.6 | -0.212 | 129.4 |
| [10554, 5732, 6141] | 0 | 31 | 145.3 | +0.146 | 142.1 |
| [6890, 2909, 7427] | 11143 | 32 | 140.2 | +0.036 | 139.5 |
| [6076, 2836, 7510] | 0 | 58 | 134.9 | -0.116 | 136.8 |
| [5286, 2950, 7499] | 0 | 40 | 160.3 | +0.437 | 148.8 |
| [4353, 3038, 7530] | 0 | 33 | 157.0 | +0.156 | 153.1 |
| [3565, 3093, 7361] | 0 | 36 | 160.7 | +0.345 | 152.4 |
| [2658, 1178, 6078] | 0 | 30 | 167.6 | +0.367 | 159.4 |

`main_s1` (crop indices [0, 43, 87, 130, 174, 217, 261, 304, 348, 391, 435, 478]):

| bbox min zyx | mask voxels | stored pts in crop | raw CT at pts | z-score at pts | raw CT crop mean |
|---|---|---|---|---|---|
| [7262, 2486, 7171] | 31113 | 101 | +0.000 | +0.000 | +0.000 |
| [8021, 2562, 7208] | 0 | 80 | +0.000 | +0.000 | +0.000 |
| [8746, 2692, 7115] | 0 | 43 | +0.000 | +0.000 | +0.000 |
| [9581, 2334, 7067] | 50094 | 141 | +0.000 | +0.000 | +0.000 |
| [9951, 5373, 6336] | 0 | 224 | +0.000 | +0.000 | +0.000 |
| [10554, 5732, 6141] | 0 | 31 | +0.000 | +0.000 | +0.000 |
| [6890, 2909, 7427] | 11143 | 32 | +0.000 | +0.000 | +0.000 |
| [6076, 2836, 7510] | 0 | 58 | +0.000 | +0.000 | +0.000 |
| [5286, 2950, 7499] | 0 | 40 | +0.000 | +0.000 | +0.000 |
| [4353, 3038, 7530] | 0 | 33 | +0.000 | +0.000 | +0.000 |
| [3565, 3093, 7361] | 0 | 36 | +0.000 | +0.000 | +0.000 |
| [2658, 1178, 6078] | 0 | 30 | +0.000 | +0.000 | +0.000 |

`branch_s1` (crop indices [0, 16, 33, 49, 65, 81, 98, 114, 130, 146, 163, 179]):

| bbox min zyx | mask voxels | stored pts in crop | raw CT at pts | z-score at pts | raw CT crop mean |
|---|---|---|---|---|---|
| [3627, 1177, 3519] | 0 | 319 | 141.0 | +0.088 | 138.3 |
| [3973, 1216, 3543] | 0 | 250 | 140.5 | +0.463 | 113.8 |
| [4385, 1272, 3481] | 0 | 162 | 133.3 | -0.348 | 139.3 |
| [4740, 1037, 3464] | 0 | 371 | 143.7 | +0.169 | 140.6 |
| [4967, 2686, 3126] | 0 | 845 | 129.3 | +0.159 | 121.3 |
| [5328, 2815, 3016] | 0 | 180 | 147.3 | +0.656 | 103.5 |
| [3351, 1395, 3657] | 0 | 133 | 150.1 | +0.328 | 143.3 |
| [2932, 1382, 3697] | 0 | 164 | 133.9 | -0.359 | 140.7 |
| [2549, 1433, 3700] | 0 | 127 | 153.3 | +0.188 | 148.8 |
| [2174, 1450, 3701] | 0 | 123 | 156.6 | +0.171 | 152.4 |
| [1702, 1471, 3635] | 0 | 136 | 155.4 | +0.011 | 155.1 |
| [1321, 545, 3014] | 0 | 125 | 161.3 | +0.094 | 159.3 |

## 5. Existing tests under neural_tracing, main vs branch

Command (identical for both, each in its own clean worktree: main = `5432721`, branch = `d6e37cb`):

```
cd <worktree>/vesuvius
PYTHONPATH=src:<worktree>:<worktree>/lasagna python -m pytest -q -p no:cacheprovider --continue-on-collection-errors \
    tests/neural_tracing src/vesuvius/neural_tracing -rfE
```

Verbatim summary:

```
== main
1 failed, 708 passed, 3 skipped, 754 warnings, 2 errors in 247.52s (0:04:07)
ERROR tests/neural_tracing/test_autoreg_mesh_model_smoke.py
ERROR tests/neural_tracing/test_autoreg_mesh_serialization.py
FAILED tests/neural_tracing/test_fiber_trace_3d.py::test_3d_fiber_infer_writes_lasagna_presence_normal_products
== branch
1 failed, 709 passed, 3 skipped, 754 warnings, 2 errors in 244.60s (0:04:04)
ERROR tests/neural_tracing/test_autoreg_mesh_model_smoke.py
ERROR tests/neural_tracing/test_autoreg_mesh_serialization.py
FAILED tests/neural_tracing/test_fiber_trace_3d.py::test_3d_fiber_infer_writes_lasagna_presence_normal_products
== new on branch:
      2 E   ModuleNotFoundError: No module named 'skimage'
      2 E   ModuleNotFoundError: No module named 'tensorstore'
```

("new on branch:" is empty. The two lines after it are the `ModuleNotFoundError` counts from the main log.) There are no new failures. The +1 pass is the new test.
The 2 errors and 1 failure are environment-only (missing `scikit-image`, `tensorstore`) and identical on both sides.

## 6. Environment

- Linux 6.18.44 x86_64, 4 vCPU, no GPU. git 2.43.0, uv 0.11.32.
- Isolated venv in the session scratchpad (the repo and its lockfile were not modified). Python 3.14.6. Packages pinned to
  `vesuvius/uv.lock` versions where the lock has them: torch 2.12.1 (PyPI cu130 wheel, run on CPU), torchvision 0.27.1,
  numpy 2.4.6, zarr 3.2.1, numcodecs 0.16.5, scipy 1.18.0, timm 1.0.27, numba 0.66.0, pandas 3.0.3, tifffile 2026.6.1, ...
  Not lock-pinned: opencv-python-headless 5.0.0.93, pytest 9.1.1. Not installed: scikit-image, tensorstore,
  volume-cartographer (C++). `uv sync` was not used. download.pytorch.org and releases.astral.sh are blocked here, and
  `vesuvius` was used via `PYTHONPATH=src`, not installed.

## 7. Doubts / limitations

1. **zarr 3 incompatibility (pre-existing, not fixed):** infer_streamline writes its output groups with
   `Group.create_dataset` (25 call sites in `inference/`). The locked zarr 3.2.1 has no such method, so
   `_run_streamline_iteration` raises `AttributeError` at `_store_iteration_geometry` on main and on this branch alike. The harness
   adds a shim (`create_dataset` → `create_array`) for that call only. You presumably run zarr 2.x. This is out of scope.
2. **Surface-mask channel is mostly empty (pre-existing, made worse by larger crops):** `upsample_voxelize_tifxyz_surface_in_bboxes`
   uses `strict_valid=True`, so any invalid tifxyz vertex in a crop's grid region empties that crop's mask. For w084 this leaves
   67/479 crops with a mask at s = 0 and 3/180 at s = 1. A level-1 crop covers 8× the level-0 volume, so it hits holes
   more often. The fix gets the CT and the mask into the same frame, but on this segment the model would still see an empty
   mask channel in most crops. I did not change this.
3. **Units of the remaining distance knobs:** `INTEGRATION_STEP_SIZE` (6), `SURFACE_ATTRACT_MAX_CORRECTION` (8) and the
   edge-support offsets now act in level-s voxels, the model frame and the frame of the predicted `surface_attract`
   vectors. Only `TIFXYZ_VOXEL_STEP`, which is derived from level-0 tifxyz metadata, is converted. If these knobs were tuned
   in level-0 units, their physical size doubles at s = 1. I could not check this without a checkpoint.
4. Output tifxyz: the merged grid is scaled back to level 0 (exact for powers of two). `save_tifxyz` metadata (`step_size`,
   voxel size) stays level-0 and consistent. The integration attrs inside the temporary zarr are in level-s units.
5. The volume from the ash2txt server named in the task could not be reached. The S3 copy was used instead (see §4).
6. Only 12 crops per run were read for CT values. Crop placement covers all crops.
7. The prediction-quality effect is not measured (no public streamline checkpoint).

## Appendix A: harness `crop_evidence.py` (verbatim)

```python
"""Crop-placement evidence for infer_streamline on real Scroll 5 data.

Runs the real `_run_streamline_iteration` of the given checkout up to the model
call (run_rowcol_bbox_inference is intercepted), then reads CT for a subset of
the crops from a local partial mirror of the public S3 zarr. Chunks are fetched
on demand; absent S3 keys (masked/empty chunks) count as fill value 0.

usage: python -I crop_evidence.py <vesuvius_src> <tifxyz_dir> <mirror_dir> <scale> <n_ct_crops> <out_json>
"""
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

src, seg, mirror, scale, n_ct, out_json = sys.argv[1:7]
sys.path.insert(0, src)
import numpy as np  # noqa: E402
import zarr  # noqa: E402

if not hasattr(zarr.Group, "create_dataset"):
    # Harness-only shim: infer_streamline uses the zarr-2 Group.create_dataset API,
    # which the locked zarr 3.2.1 no longer provides. Only used for geometry attrs here.
    zarr.Group.create_dataset = lambda self, name, *, data, chunks=None, **kw: self.create_array(
        name, data=data, chunks=chunks if chunks is not None else "auto")
from vesuvius.neural_tracing.inference import infer_streamline as m  # noqa: E402
from vesuvius.neural_tracing.datasets.common import _read_volume_crop  # noqa: E402

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
VOL = "PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr"
CONFIG = Path(src) / "vesuvius/neural_tracing/configs/config_rowcol_cond_dense.json"
scale = int(scale)
mirror = Path(mirror)
stats = {"fetched_chunks": 0, "absent_chunks": 0, "fetched_bytes": 0}


def fetch(key):
    dst = mirror / key
    if dst.exists() or (mirror / (key + ".absent")).exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = urllib.request.urlopen(f"{BUCKET}/{VOL}/{key}", timeout=120).read()
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        (mirror / (key + ".absent")).touch()
        stats["absent_chunks"] += 1
        return
    dst.write_bytes(data)
    stats["fetched_chunks"] += 1
    stats["fetched_bytes"] += len(data)


for key in (".zgroup", ".zattrs", f"{scale}/.zarray"):
    fetch(key)
zarray = json.loads((mirror / f"{scale}/.zarray").read_text())
shape = np.asarray(zarray["shape"], dtype=np.int64)
chunks = np.asarray(zarray["chunks"], dtype=np.int64)


def fetch_region(min_corner, max_corner):
    lo = np.maximum(min_corner, 0)
    hi = np.minimum(max_corner, shape)
    if np.any(hi <= lo):
        return
    c0, c1 = lo // chunks, (hi - 1) // chunks
    for cz in range(c0[0], c1[0] + 1):
        for cy in range(c0[1], c1[1] + 1):
            for cx in range(c0[2], c1[2] + 1):
                fetch(f"{scale}/{cz}/{cy}/{cx}")


crop_size = m._crop_size_from_config(json.loads(CONFIG.read_text()))
m.VOLUME_SCALE = scale
m.VOLUME_PATH = str(mirror)
m.TIFXYZ_PATH = str(seg)
m.TIFXYZ_VOXEL_STEP = m.derive_tifxyz_voxel_step(seg)
m.DEVICE = "cpu"
m.RUN_OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="crop_evidence_"))
m.RUN_TIMESTAMP = "evidence"
volume_array = m._open_volume_array(str(mirror), scale)

captured = {}


class _Stop(Exception):
    pass


def _capture(model, model_config, volume_array, surface, bboxes, crop_size, cond_direction, zarr_root,
             stored_zyxs, valid, **kwargs):
    captured.update(surface=surface, bboxes=bboxes, cond_direction=cond_direction,
                    stored_zyxs=stored_zyxs, valid=valid)
    raise _Stop


m.run_rowcol_bbox_inference = _capture
try:
    m._run_streamline_iteration(None, {}, volume_array, crop_size, seg, 1, 1, m.RUN_OUTPUT_DIR / "tifxyz")
except _Stop:
    pass

bboxes = captured["bboxes"]
mins = np.asarray([m._bbox_min_corner(b) for b in bboxes], dtype=np.int64)
maxs = mins + np.asarray(crop_size, dtype=np.int64)
fully_inside = np.all((mins >= 0) & (maxs <= shape), axis=1)
fully_outside = np.any((maxs <= 0) | (mins >= shape), axis=1)

ct_idx = np.unique(np.linspace(0, len(bboxes) - 1, int(n_ct)).round().astype(int))
subset = [bboxes[i] for i in ct_idx]
for b in subset:
    fetch_region(m._bbox_min_corner(b), m._bbox_min_corner(b) + np.asarray(crop_size))
vox = m.upsample_voxelize_tifxyz_surface_in_bboxes(
    captured["surface"], subset, crop_size,
    stored_zyxs=captured["stored_zyxs"], valid=captured["valid"],
)
inputs = m._prepare_rowcol_inputs(volume_array, vox, crop_size, captured["cond_direction"]).numpy()
pts_all = np.asarray(captured["stored_zyxs"])[captured["valid"]]
per_crop = []
for b, item, inp in zip(subset, vox, inputs):
    mn = m._bbox_min_corner(b)
    raw = _read_volume_crop(volume_array, crop_size, mn, mn + np.asarray(crop_size), "unit") * 255.0
    mask = inp[1] > 0
    # Stored tifxyz points (in the coordinates the crops were built from) inside this crop.
    loc = np.floor(pts_all - mn).astype(np.int64)
    inside = np.all((loc >= 0) & (loc < np.asarray(crop_size)), axis=1)
    li = tuple(loc[inside].T)
    per_crop.append({
        "bbox": [int(v) for v in b],
        "surface_voxels": int(mask.sum()),
        "raw_mean_at_surface": float(raw[mask].mean()) if mask.any() else None,
        "zscore_mean_at_surface": float(inp[0][mask].mean()) if mask.any() else None,
        "stored_points_in_crop": int(inside.sum()),
        "raw_mean_at_points": float(raw[li].mean()) if inside.any() else None,
        "zscore_mean_at_points": float(inp[0][li].mean()) if inside.any() else None,
        "raw_mean_crop": float(raw.mean()),
        "raw_nonzero_frac": float((raw > 0).mean()),
    })


def _mean(key):
    vals = [c[key] for c in per_crop if c[key] is not None]
    return float(np.mean(vals)) if vals else None


h = hashlib.sha256()
h.update(np.asarray(bboxes, dtype=np.int64).tobytes())
h.update(np.ascontiguousarray(inputs).tobytes())
result = {
    "src": src,
    "tifxyz": str(seg),
    "volume": f"{BUCKET}/{VOL}",
    "volume_scale": scale,
    "level_shape_zyx": shape.tolist(),
    "crop_size": list(crop_size),
    "tifxyz_voxel_step": float(m.TIFXYZ_VOXEL_STEP),
    "edge_axis_scale_surface_scale": [float(v) for v in captured["surface"]._scale],
    "stored_zyx_min": np.asarray(captured["stored_zyxs"])[captured["valid"]].min(axis=0).tolist(),
    "stored_zyx_max": np.asarray(captured["stored_zyxs"])[captured["valid"]].max(axis=0).tolist(),
    "n_crops": len(bboxes),
    "crops_fully_inside": int(fully_inside.sum()),
    "crops_not_fully_inside": int((~fully_inside).sum()),
    "crops_fully_outside": int(fully_outside.sum()),
    "share_not_fully_inside": float((~fully_inside).mean()),
    "share_fully_outside": float(fully_outside.mean()),
    "ct_crop_indices": ct_idx.tolist(),
    "ct_crops_with_surface_voxels": sum(1 for c in per_crop if c["surface_voxels"]),
    "mean_raw_at_surface": _mean("raw_mean_at_surface"),
    "mean_raw_crop": _mean("raw_mean_crop"),
    "mean_raw_nonzero_frac": _mean("raw_nonzero_frac"),
    "mean_zscore_at_surface": _mean("zscore_mean_at_surface"),
    "ct_crops_with_stored_points": sum(1 for c in per_crop if c["stored_points_in_crop"]),
    "mean_raw_at_points": _mean("raw_mean_at_points"),
    "mean_zscore_at_points": _mean("zscore_mean_at_points"),
    "sha256_bboxes_and_inputs": h.hexdigest(),
    "download": stats,
    "per_crop": per_crop,
}
Path(out_json).write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k != "per_crop"}, indent=1))
```

## Appendix B: harness `mask_coverage.py` (verbatim)

```python
"""Surface-mask coverage (strict voxelization, as used by run_rowcol_bbox_inference) for infer_streamline on real Scroll 5 data.

Runs the real `_run_streamline_iteration` of the given checkout up to the model
call (run_rowcol_bbox_inference is intercepted), then reads CT for a subset of
the crops from a local partial mirror of the public S3 zarr. Chunks are fetched
on demand; absent S3 keys (masked/empty chunks) count as fill value 0.

usage: python -I crop_evidence.py <vesuvius_src> <tifxyz_dir> <mirror_dir> <scale> <n_ct_crops> <out_json>
"""
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

src, seg, mirror, scale, n_ct, out_json = sys.argv[1:7]
sys.path.insert(0, src)
import numpy as np  # noqa: E402
import zarr  # noqa: E402

if not hasattr(zarr.Group, "create_dataset"):
    # Harness-only shim: infer_streamline uses the zarr-2 Group.create_dataset API,
    # which the locked zarr 3.2.1 no longer provides. Only used for geometry attrs here.
    zarr.Group.create_dataset = lambda self, name, *, data, chunks=None, **kw: self.create_array(
        name, data=data, chunks=chunks if chunks is not None else "auto")
from vesuvius.neural_tracing.inference import infer_streamline as m  # noqa: E402
from vesuvius.neural_tracing.datasets.common import _read_volume_crop  # noqa: E402

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
VOL = "PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr"
CONFIG = Path(src) / "vesuvius/neural_tracing/configs/config_rowcol_cond_dense.json"
scale = int(scale)
mirror = Path(mirror)
stats = {"fetched_chunks": 0, "absent_chunks": 0, "fetched_bytes": 0}


def fetch(key):
    dst = mirror / key
    if dst.exists() or (mirror / (key + ".absent")).exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = urllib.request.urlopen(f"{BUCKET}/{VOL}/{key}", timeout=120).read()
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        (mirror / (key + ".absent")).touch()
        stats["absent_chunks"] += 1
        return
    dst.write_bytes(data)
    stats["fetched_chunks"] += 1
    stats["fetched_bytes"] += len(data)


for key in (".zgroup", ".zattrs", f"{scale}/.zarray"):
    fetch(key)
zarray = json.loads((mirror / f"{scale}/.zarray").read_text())
shape = np.asarray(zarray["shape"], dtype=np.int64)
chunks = np.asarray(zarray["chunks"], dtype=np.int64)


def fetch_region(min_corner, max_corner):
    lo = np.maximum(min_corner, 0)
    hi = np.minimum(max_corner, shape)
    if np.any(hi <= lo):
        return
    c0, c1 = lo // chunks, (hi - 1) // chunks
    for cz in range(c0[0], c1[0] + 1):
        for cy in range(c0[1], c1[1] + 1):
            for cx in range(c0[2], c1[2] + 1):
                fetch(f"{scale}/{cz}/{cy}/{cx}")


crop_size = m._crop_size_from_config(json.loads(CONFIG.read_text()))
m.VOLUME_SCALE = scale
m.VOLUME_PATH = str(mirror)
m.TIFXYZ_PATH = str(seg)
m.TIFXYZ_VOXEL_STEP = m.derive_tifxyz_voxel_step(seg)
m.DEVICE = "cpu"
m.RUN_OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="crop_evidence_"))
m.RUN_TIMESTAMP = "evidence"
volume_array = m._open_volume_array(str(mirror), scale)

captured = {}


class _Stop(Exception):
    pass


def _capture(model, model_config, volume_array, surface, bboxes, crop_size, cond_direction, zarr_root,
             stored_zyxs, valid, **kwargs):
    captured.update(surface=surface, bboxes=bboxes, cond_direction=cond_direction,
                    stored_zyxs=stored_zyxs, valid=valid)
    raise _Stop


m.run_rowcol_bbox_inference = _capture
try:
    m._run_streamline_iteration(None, {}, volume_array, crop_size, seg, 1, 1, m.RUN_OUTPUT_DIR / "tifxyz")
except _Stop:
    pass

bboxes = captured["bboxes"]
vox = m.upsample_voxelize_tifxyz_surface_in_bboxes(captured["surface"], bboxes, crop_size,
    stored_zyxs=captured["stored_zyxs"], valid=captured["valid"])
n = [int(v["voxels"].sum()) for v in vox]
r = {"volume_scale": scale, "n_crops": len(bboxes), "crops_with_mask_strict": sum(1 for x in n if x), "mask_voxels_strict_total": sum(n)}
Path(out_json).write_text(json.dumps(r, indent=1)); print(json.dumps(r))
```
