import sys
from pathlib import Path

import pytest
import torch


REPO_DIR = Path(__file__).resolve().parents[1]
NANOGPT_DIR = REPO_DIR / "examples" / "nanogpt"
if str(NANOGPT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOGPT_DIR))

from attnres import BlockAttnResReference, FullAttnResReference  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def _attnres_config(**overrides) -> GPTConfig:
    kwargs = dict(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
        hc_num_streams=1,
        hc_num_fracs=1,
        hc_disable=True,
        mhc=False,
        v_residual=False,
        attnres_variant="full",
    )
    kwargs.update(overrides)
    return GPTConfig(**kwargs)


def _copy_attnres_queries(target: GPT, source: GPT) -> None:
    with torch.no_grad():
        target.attnres_mixer.mixer.queries.copy_(source.attnres_mixer.mixer.queries)


def test_full_attnres_gpt_matches_reference_rollout() -> None:
    torch.manual_seed(0)
    model = GPT(_attnres_config())
    idx = torch.randint(0, model.config.vocab_size, (2, 5))
    targets = torch.randint(0, model.config.vocab_size, (2, 5))

    with torch.no_grad():
        model.attnres_mixer.mixer.queries.copy_(
            torch.randn_like(model.attnres_mixer.mixer.queries)
        )

    reference = FullAttnResReference(
        num_layers=model.attnres_num_logical_layers,
        dim=model.config.n_embd,
        eps=model.config.attnres_eps,
    )
    with torch.no_grad():
        reference.mixer.queries.copy_(model.attnres_mixer.mixer.queries)

    logits, loss = model(idx, targets)

    embedding = model._embed_input(idx)
    logical_outputs: list[torch.Tensor] = []
    for block_index, block in enumerate(model.transformer["h"]):
        attn_input = reference.layer_input(
            embedding=embedding,
            prior_layer_outputs=logical_outputs,
            layer_index=2 * block_index,
        )
        logical_outputs.append(block.attn_output(attn_input, vrl_state=None))

        mlp_input = reference.layer_input(
            embedding=embedding,
            prior_layer_outputs=logical_outputs,
            layer_index=2 * block_index + 1,
        )
        logical_outputs.append(block.mlp_output(mlp_input))

    hidden = reference.final_output(embedding=embedding, layer_outputs=logical_outputs)
    hidden = model.transformer["ln_f"](hidden)
    logits_ref = model.lm_head(hidden)
    loss_ref = torch.nn.functional.cross_entropy(
        logits_ref.view(-1, logits_ref.size(-1)),
        targets.view(-1),
    )

    assert torch.allclose(logits, logits_ref, atol=1e-7, rtol=0.0)
    assert torch.allclose(loss, loss_ref, atol=1e-7, rtol=0.0)


def test_full_attnres_allocates_queries_per_logical_layer_plus_output() -> None:
    config = _attnres_config(n_layer=3)
    model = GPT(config)

    assert model.attnres_num_logical_layers == 6
    assert model.attnres_mixer.mixer.queries.shape == (7, config.n_embd)


def test_block_attnres_gpt_matches_reference_rollout() -> None:
    torch.manual_seed(0)
    model = GPT(_attnres_config(attnres_variant="block", attnres_block_size=2))
    idx = torch.randint(0, model.config.vocab_size, (2, 5))
    targets = torch.randint(0, model.config.vocab_size, (2, 5))

    with torch.no_grad():
        model.attnres_mixer.mixer.queries.copy_(
            torch.randn_like(model.attnres_mixer.mixer.queries)
        )

    reference = BlockAttnResReference(
        num_layers=model.attnres_num_logical_layers,
        dim=model.config.n_embd,
        block_size=model.config.attnres_block_size,
        eps=model.config.attnres_eps,
    )
    with torch.no_grad():
        reference.mixer.queries.copy_(model.attnres_mixer.mixer.queries)

    logits, loss = model(idx, targets)

    embedding = model._embed_input(idx)
    state = reference.init_state(embedding)
    for block_index, block in enumerate(model.transformer["h"]):
        attn_input = reference.layer_input(state=state, layer_index=2 * block_index)
        attn_output = block.attn_output(attn_input, vrl_state=None)
        reference.append_layer_output(state, attn_output)

        mlp_input = reference.layer_input(state=state, layer_index=2 * block_index + 1)
        mlp_output = block.mlp_output(mlp_input)
        reference.append_layer_output(state, mlp_output)

    hidden = reference.final_output(state)
    hidden = model.transformer["ln_f"](hidden)
    logits_ref = model.lm_head(hidden)
    loss_ref = torch.nn.functional.cross_entropy(
        logits_ref.view(-1, logits_ref.size(-1)),
        targets.view(-1),
    )

    assert torch.allclose(logits, logits_ref, atol=1e-7, rtol=0.0)
    assert torch.allclose(loss, loss_ref, atol=1e-7, rtol=0.0)


def test_block_attnres_block_size_one_matches_full_attnres_model_exactly() -> None:
    torch.manual_seed(0)
    full_model = GPT(_attnres_config(attnres_variant="full"))
    block_model = GPT(_attnres_config(attnres_variant="block", attnres_block_size=1))
    block_model.load_state_dict(full_model.state_dict(), strict=False)
    _copy_attnres_queries(block_model, full_model)

    idx = torch.randint(0, full_model.config.vocab_size, (2, 5))
    targets = torch.randint(0, full_model.config.vocab_size, (2, 5))

    full_logits, full_loss = full_model(idx, targets)
    block_logits, block_loss = block_model(idx, targets)

    assert torch.allclose(full_logits, block_logits, atol=1e-7, rtol=0.0)
    assert torch.allclose(full_loss, block_loss, atol=1e-7, rtol=0.0)


def test_attnres_rejects_mhc_combo() -> None:
    with pytest.raises(ValueError, match="cannot be combined with mhc"):
        GPT(_attnres_config(mhc=True))


def test_attnres_requires_hyper_connections_disabled() -> None:
    with pytest.raises(ValueError, match="requires hc_disable=True"):
        GPT(_attnres_config(hc_disable=False))


def test_attnres_rejects_nonpositive_block_size() -> None:
    with pytest.raises(ValueError, match="attnres_block_size must be positive"):
        GPT(_attnres_config(attnres_variant="block", attnres_block_size=0))
