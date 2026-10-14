import torch

from vesuvius.models.run import inference
from vesuvius.models.run.inference import Inferer


def test_trainpy_zscore_with_intensity_properties_uses_instance_zscore(monkeypatch):
    captured = {}

    class FakeDataset:
        collate_fn = staticmethod(lambda batch: batch)

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.all_positions = []

        def __len__(self):
            return 0

    monkeypatch.setattr(inference, 'VCDataset', FakeDataset)

    inferer = Inferer.__new__(Inferer)
    # As loaded from a train.py checkpoint trained with --no-skip-intensity-sampling
    inferer.model_normalization_scheme = 'zscore'
    inferer.model_intensity_properties = {'mean': 120.5, 'std': 30.25}
    inferer.normalization_scheme = 'instance_zscore'
    for name, value in dict(
            input='unused.zarr', patch_size=(8, 8, 8), overlap=0.5, num_parts=1,
            part_id=0, input_format='zarr', verbose=False, skip_empty_patches=False,
            scroll_id=None, segment_id=None, energy=None, resolution=None,
            input_anon=False, bbox=None, read_retries=1, chunk_cache_mb=0,
            device=torch.device('cpu'), max_patches=None).items():
        setattr(inferer, name, value)

    inferer._create_dataset_and_loader()

    assert captured['normalization_scheme'] == 'instance_zscore'
    assert captured['global_mean'] is None
    assert captured['global_std'] is None
