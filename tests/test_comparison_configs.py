from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "examples" / "nanogpt" / "config"


def load_config(name: str) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec((CONFIG_DIR / name).read_text(encoding="utf-8"), namespace)
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


def test_48l_comparison_family_shares_model_and_budget() -> None:
    configs = {
        "baseline": load_config("train_fineweb10B_48l.py"),
        "mhc": load_config("train_fineweb10B_mhc_48l.py"),
        "attnres_full": load_config("train_fineweb10B_attnres_full_48l.py"),
        "attnres_block": load_config("train_fineweb10B_attnres_block_48l.py"),
    }

    shared_keys = {
        "dataset": "fineweb10B",
        "block_size": 1024,
        "n_layer": 48,
        "n_head": 6,
        "n_embd": 150,
        "dropout": 0.0,
        "bias": False,
        "batch_size": 8,
        "gradient_accumulation_steps": 16,
        "target_tokens_per_iter": 131072,
        "target_tokens": 1048576000,
        "max_iters": 8000,
        "eval_interval": 500,
        "log_interval": 10,
        "eval_iters": 100,
        "learning_rate": 6e-4,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "warmup_iters": 200,
        "lr_decay_iters": 8000,
        "lock_lr_decay_to_max_iters": True,
        "min_lr": 6e-5,
        "dtype": "bfloat16",
    }

    for config in configs.values():
        for key, expected in shared_keys.items():
            assert config[key] == expected


def test_48l_comparison_family_mechanism_flags() -> None:
    baseline = load_config("train_fineweb10B_48l.py")
    mhc = load_config("train_fineweb10B_mhc_48l.py")
    attnres_full = load_config("train_fineweb10B_attnres_full_48l.py")
    attnres_block = load_config("train_fineweb10B_attnres_block_48l.py")

    assert baseline["hc_disable"] is True
    assert baseline.get("mhc", False) is False
    assert baseline.get("attnres_variant", "none") == "none"

    assert mhc["hc_disable"] is False
    assert mhc["mhc"] is True
    assert mhc["hc_num_streams"] == 4
    assert mhc["sinkhorn_iters"] == 20

    assert attnres_full["hc_disable"] is True
    assert attnres_full["mhc"] is False
    assert attnres_full["attnres_variant"] == "full"

    assert attnres_block["hc_disable"] is True
    assert attnres_block["mhc"] is False
    assert attnres_block["attnres_variant"] == "block"
    assert attnres_block["attnres_block_size"] == 4
