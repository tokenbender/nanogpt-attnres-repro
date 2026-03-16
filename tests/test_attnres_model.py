import sys
from pathlib import Path

import pytest
import torch


REPO_DIR = Path(__file__).resolve().parents[1]
NANOGPT_DIR = REPO_DIR / "examples" / "nanogpt"
if str(NANOGPT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOGPT_DIR))

from attnres import FullAttnResReference  # noqa: E402
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


def test_attnres_rejects_mhc_combo() -> None:
    with pytest.raises(ValueError, match="cannot be combined with mhc"):
        GPT(_attnres_config(mhc=True))


def test_attnres_requires_hyper_connections_disabled() -> None:
    with pytest.raises(ValueError, match="requires hc_disable=True"):
        GPT(_attnres_config(hc_disable=False))


def test_block_attnres_training_path_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError, match="not integrated"):
        GPT(_attnres_config(attnres_variant="block"))
