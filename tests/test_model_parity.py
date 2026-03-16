from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO_DIR = Path(__file__).resolve().parents[1]
LOCAL_ROOT = REPO_DIR
LOCAL_NANOGPT_DIR = LOCAL_ROOT / "examples" / "nanogpt"
SOURCE_ROOT = Path(
    "/Users/tokenbender/Documents/mHC-manifold-constrained-hyper-connections"
)
SOURCE_NANOGPT_DIR = SOURCE_ROOT / "examples" / "nanogpt"


def _purge_imports() -> None:
    for name in list(sys.modules):
        if name == "hyper_connections" or name.startswith("hyper_connections."):
            sys.modules.pop(name)
        elif name == "value_residual":
            sys.modules.pop(name)


def _load_module(root: Path, nanogpt_dir: Path, module_name: str):
    _purge_imports()
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(nanogpt_dir))
        spec = importlib.util.spec_from_file_location(
            module_name, nanogpt_dir / "model.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path
        _purge_imports()


def _model_config_kwargs(**overrides):
    kwargs = dict(
        block_size=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        hc_num_streams=1,
        hc_num_fracs=1,
        hc_disable=True,
        mhc=False,
        v_residual=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_baseline_block_matches_manual_pre_norm_residual() -> None:
    local_model = _load_module(LOCAL_ROOT, LOCAL_NANOGPT_DIR, "local_nanogpt_model")
    init_hc, _, _ = local_model.get_init_and_expand_reduce_stream_functions(
        1,
        num_fracs=1,
        disable=True,
    )
    config = local_model.GPTConfig(**_model_config_kwargs())
    block = local_model.Block(config, layer_idx=0, init_hc=init_hc)
    x = torch.randn(2, 16, 16)

    with torch.no_grad():
        manual = x + block.attn(block.ln_1(x))
        manual = manual + block.mlp(block.ln_2(manual))
        actual = block(x)

    assert torch.allclose(actual, manual, atol=1e-7, rtol=0.0)


def test_vendored_baseline_gpt_matches_source_repo() -> None:
    torch.manual_seed(0)
    local_model = _load_module(
        LOCAL_ROOT, LOCAL_NANOGPT_DIR, "local_nanogpt_model_baseline"
    )
    source_model = _load_module(
        SOURCE_ROOT, SOURCE_NANOGPT_DIR, "source_nanogpt_model_baseline"
    )

    local = local_model.GPT(local_model.GPTConfig(**_model_config_kwargs()))
    source = source_model.GPT(source_model.GPTConfig(**_model_config_kwargs()))
    source.load_state_dict(local.state_dict())

    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))

    local_logits, local_loss = local(idx, targets)
    source_logits, source_loss = source(idx, targets)

    assert torch.allclose(local_logits, source_logits, atol=1e-7, rtol=0.0)
    assert torch.allclose(local_loss, source_loss, atol=1e-7, rtol=0.0)


def test_vendored_mhc_gpt_matches_source_repo() -> None:
    torch.manual_seed(0)
    local_model = _load_module(LOCAL_ROOT, LOCAL_NANOGPT_DIR, "local_nanogpt_model_mhc")
    source_model = _load_module(
        SOURCE_ROOT, SOURCE_NANOGPT_DIR, "source_nanogpt_model_mhc"
    )

    cfg_kwargs = _model_config_kwargs(
        hc_num_streams=4,
        hc_disable=False,
        mhc=True,
        sinkhorn_iters=10,
        sinkhorn_tau=0.05,
        mhc_h_res_proj="sinkhorn",
        ns_steps=5,
        ns_eps=1e-7,
        ns_coeffs=(3.0, -3.2, 1.2),
        mhc_residual_identity_mix=False,
        mhc_residual_alpha=0.01,
    )
    local = local_model.GPT(local_model.GPTConfig(**cfg_kwargs))
    source = source_model.GPT(source_model.GPTConfig(**cfg_kwargs))
    source.load_state_dict(local.state_dict())

    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))

    local_logits, local_loss = local(idx, targets)
    source_logits, source_loss = source(idx, targets)

    assert torch.allclose(local_logits, source_logits, atol=1e-7, rtol=0.0)
    assert torch.allclose(local_loss, source_loss, atol=1e-7, rtol=0.0)
