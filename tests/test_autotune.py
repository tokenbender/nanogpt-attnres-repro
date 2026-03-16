import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
NANOGPT_DIR = REPO_DIR / "examples" / "nanogpt"
if str(NANOGPT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOGPT_DIR))

from autotune import (  # noqa: E402
    default_batch_sizes,
    derive_target_tokens_per_iter,
    derive_total_accumulation,
    is_legal_batch_size,
    load_config_values,
)


def test_derive_target_tokens_per_iter_prefers_explicit_target() -> None:
    cfg = {
        "batch_size": 32,
        "block_size": 1024,
        "gradient_accumulation_steps": 4,
        "target_tokens_per_iter": 131072,
    }
    assert derive_target_tokens_per_iter(cfg) == 131072


def test_derive_target_tokens_per_iter_falls_back_to_batch_times_seq_times_accum() -> None:
    cfg = {
        "batch_size": 8,
        "block_size": 256,
        "gradient_accumulation_steps": 3,
        "target_tokens_per_iter": None,
    }
    assert derive_target_tokens_per_iter(cfg) == 6144


def test_derive_total_accumulation_returns_none_when_incompatible() -> None:
    assert (
        derive_total_accumulation(
            batch_size=48,
            block_size=1024,
            target_tokens_per_iter=131072,
        )
        is None
    )


def test_is_legal_batch_size_checks_world_size_divisibility() -> None:
    assert is_legal_batch_size(
        batch_size=32,
        block_size=1024,
        target_tokens_per_iter=131072,
        world_size=4,
    )
    assert not is_legal_batch_size(
        batch_size=64,
        block_size=1024,
        target_tokens_per_iter=131072,
        world_size=4,
    )


def test_default_batch_sizes_prefers_legal_power_of_two_candidates() -> None:
    candidates = default_batch_sizes(
        base_batch_size=32,
        block_size=1024,
        target_tokens_per_iter=131072,
        world_size=1,
    )
    assert candidates == [1, 2, 4, 8, 16, 32, 64, 128]


def test_load_config_values_applies_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "toy_config.py"
    config_path.write_text(
        "batch_size = 32\nblock_size = 1024\ngradient_accumulation_steps = 4\n",
        encoding="utf-8",
    )
    loaded = load_config_values(config_path, ["batch_size=64", "target_tokens=123"])
    assert loaded["batch_size"] == 64
    assert loaded["block_size"] == 1024
    assert loaded["gradient_accumulation_steps"] == 4
    assert loaded["target_tokens"] == 123
