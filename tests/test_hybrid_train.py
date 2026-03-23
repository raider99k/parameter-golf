import torch

from hybrid_golf.data import TokenBatcher, write_data_shard
from hybrid_golf.train import _schedule_progress_frac


def test_schedule_progress_uses_step_progress_without_wallclock():
    progress = _schedule_progress_frac(step=3, total_steps=10, elapsed_seconds=0.0, max_wallclock_seconds=0.0)
    assert progress == 0.4


def test_schedule_progress_advances_with_wallclock_budget():
    progress = _schedule_progress_frac(
        step=1_907,
        total_steps=1_000_000,
        elapsed_seconds=102.0,
        max_wallclock_seconds=120.0,
    )
    assert 0.84 < progress < 0.86


def test_schedule_progress_prefers_farther_step_progress_when_ahead_of_wallclock():
    progress = _schedule_progress_frac(
        step=799,
        total_steps=1_000,
        elapsed_seconds=10.0,
        max_wallclock_seconds=100.0,
    )
    assert progress == 0.8


def test_token_batcher_splits_global_batch_across_grad_accum_steps(tmp_path):
    shard_path = tmp_path / "fineweb_train_000000.bin"
    write_data_shard(shard_path, torch.arange(0, 257, dtype=torch.int64))
    batcher = TokenBatcher(str(shard_path), device=torch.device("cpu"))
    x, y = batcher.next_batch(global_tokens=64, seq_len=16, grad_accum_steps=2)
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
