"""Compressed tifxyz are indexed (not skipped), and a rewritten patch file invalidates the cached index."""
import json

import numpy as np
import tifffile

from vesuvius.neural_tracing.tracers.sheets.data.patches import PatchStore, build_index


def _write_patch(patch, compression):
    """A 30 x 20 tifxyz on the plane x = 60.5 with 2 vx cells, as written by ``write_tifxyz`` (LZW) or uncompressed."""
    patch.mkdir(parents=True, exist_ok=True)
    r, c = np.meshgrid(np.arange(30), np.arange(20), indexing='ij')
    for name, values in dict(x=np.full(r.shape, 60.5), y=20.+2*c, z=20.+2*r).items():
        tifffile.imwrite(patch/f'{name}.tif', values.astype(np.float32), compression=compression)
    (patch/'meta.json').write_text(json.dumps(dict(scale=[.5, .5])))


def test_compressed_patch_is_indexed_and_readable(tmp_path, caplog):
    _write_patch(tmp_path/'patches'/'p0', 'lzw')
    index = build_index(tmp_path/'patches', tmp_path/'cache')
    assert [(i.name, i.rows, i.cols, i.valid, i.cell) for i in index] == [('p0', 30, 20, 600, 2.)]
    assert 'compressed tifxyz' in caplog.text
    x, y, z = PatchStore().arrays(index[0])
    assert float(y[3, 4]) == 28. and float(z[3, 4]) == 26.


def test_rewritten_patch_file_invalidates_cache(tmp_path):
    patches = tmp_path/'patches'
    _write_patch(patches/'p0', None)
    (patches/'p0'/'x.tif').write_bytes(b'not a tiff')  # unreadable: skipped, and the empty index is cached
    assert build_index(patches, tmp_path/'cache') == []
    _write_patch(patches/'p0', None)
    assert len(build_index(patches, tmp_path/'cache')) == 1
