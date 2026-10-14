import copy
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from vesuvius.models.training.train import BaseTrainer


def _groupnorm_stack(c=4, depth=16):
    # All-zero input -> zero variance in every GroupNorm(num_groups=C); backward overflows.
    layers = [torch.nn.Conv3d(1, c, 1)]
    for _ in range(depth):
        layers += [torch.nn.GroupNorm(num_groups=c, num_channels=c), torch.nn.Conv3d(c, c, 1)]
    model = torch.nn.Sequential(*layers)
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            torch.nn.init.ones_(m.weight)
    return model


def test_dummy_scaler_skips_step_with_nonfinite_grad_norm():
    torch.manual_seed(0)
    model, x = _groupnorm_stack(), torch.zeros(1, 1, 4, 4, 4)
    target = torch.arange(4.0).view(1, 1, 1, 1, 4)
    probe = copy.deepcopy(model)
    (probe(x) * target).mean().backward()
    assert not torch.isfinite(torch.nn.utils.clip_grad_norm_(probe.parameters(), 12.0))

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.mgr, trainer.startup_timing = SimpleNamespace(gradient_clip=12.0), False
    trainer._prepare_batch = lambda d, is_training: d
    trainer._accumulate_aug_timers = trainer._update_ema_model = lambda *a: None
    trainer._get_model_outputs = lambda m, d: (d["image"], {}, m(d["image"]))
    trainer._compute_train_loss = lambda out, t, fns: ((out * target).mean(), {})
    before = copy.deepcopy(model.state_dict())
    trainer._train_step(model, {"image": x}, None, False, nullcontext(), epoch=0, step=0,
                        scaler=trainer._get_scaler("cpu", use_amp=False),
                        optimizer=torch.optim.SGD(model.parameters(), lr=0.1), num_iters=1)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
