import math
import sys
from pathlib import Path

import torch


REPO_DIR = Path(__file__).resolve().parents[1]
NANOGPT_DIR = REPO_DIR / "examples" / "nanogpt"
if str(NANOGPT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOGPT_DIR))

from attnres import BlockAttnResReference, FullAttnResReference  # noqa: E402


def test_full_attnres_zero_queries_give_uniform_average() -> None:
    mixer = FullAttnResReference(num_layers=3, dim=2)
    embedding = torch.tensor([[1.0, 3.0]])
    layer_outputs = [torch.tensor([[5.0, 7.0]]), torch.tensor([[9.0, 11.0]])]

    mixed, weights = mixer.layer_input_with_weights(
        embedding=embedding,
        prior_layer_outputs=layer_outputs,
        layer_index=2,
    )

    expected = (embedding + layer_outputs[0] + layer_outputs[1]) / 3.0
    assert torch.allclose(mixed, expected)
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0))


def test_full_attnres_rejects_wrong_source_count() -> None:
    mixer = FullAttnResReference(num_layers=2, dim=4)
    embedding = torch.randn(2, 4)

    try:
        mixer.layer_input(embedding=embedding, prior_layer_outputs=[], layer_index=1)
    except ValueError as exc:
        assert "one prior layer output per completed logical layer" in str(exc)
    else:
        raise AssertionError("expected source-count validation to fail")


def test_full_attnres_rmsnorm_removes_pure_scale_advantage() -> None:
    mixer = FullAttnResReference(num_layers=2, dim=3)
    with torch.no_grad():
        mixer.mixer.queries[1].copy_(torch.tensor([0.25, -0.5, 1.0]))

    source = torch.tensor([[1.0, -2.0, 3.0]])
    mixed, weights = mixer.layer_input_with_weights(
        embedding=source,
        prior_layer_outputs=[10.0 * source],
        layer_index=1,
    )

    assert math.isclose(weights[0].item(), weights[1].item(), rel_tol=0.0, abs_tol=1e-7)
    expected = 0.5 * source + 0.5 * (10.0 * source)
    assert torch.allclose(mixed, expected, atol=1e-6, rtol=0.0)


def test_block_attnres_bookkeeping_matches_paper_source_sets() -> None:
    mixer = BlockAttnResReference(num_layers=4, dim=2, block_size=2)
    embedding = torch.tensor([[0.0, 1.0]])
    state = mixer.init_state(embedding)

    assert len(state.current_sources()) == 1
    assert torch.allclose(state.current_sources()[0], embedding)

    out1 = torch.tensor([[1.0, 0.0]])
    mixer.append_layer_output(state, out1)
    assert len(state.current_sources()) == 2
    assert torch.allclose(state.current_sources()[0], embedding)
    assert torch.allclose(state.current_sources()[1], out1)

    out2 = torch.tensor([[0.0, 2.0]])
    mixer.append_layer_output(state, out2)
    assert len(state.current_sources()) == 2
    assert torch.allclose(state.current_sources()[0], embedding)
    assert torch.allclose(state.current_sources()[1], out1 + out2)

    out3 = torch.tensor([[3.0, 4.0]])
    mixer.append_layer_output(state, out3)
    assert len(state.current_sources()) == 3
    assert torch.allclose(state.current_sources()[0], embedding)
    assert torch.allclose(state.current_sources()[1], out1 + out2)
    assert torch.allclose(state.current_sources()[2], out3)


def test_block_attnres_final_output_includes_incomplete_last_block() -> None:
    mixer = BlockAttnResReference(num_layers=3, dim=2, block_size=2)
    embedding = torch.tensor([[1.0, 2.0]])
    state = mixer.init_state(embedding)

    mixer.append_layer_output(state, torch.tensor([[3.0, 4.0]]))
    mixer.append_layer_output(state, torch.tensor([[5.0, 6.0]]))
    mixer.append_layer_output(state, torch.tensor([[7.0, 8.0]]))

    final_sources = state.final_sources()
    assert len(final_sources) == 3
    assert torch.allclose(final_sources[0], embedding)
    assert torch.allclose(final_sources[1], torch.tensor([[8.0, 10.0]]))
    assert torch.allclose(final_sources[2], torch.tensor([[7.0, 8.0]]))


def test_block_attnres_block_size_one_matches_full_attnres_exactly() -> None:
    num_layers = 4
    dim = 3
    full = FullAttnResReference(num_layers=num_layers, dim=dim)
    block = BlockAttnResReference(num_layers=num_layers, dim=dim, block_size=1)

    with torch.no_grad():
        queries = torch.randn(num_layers + 1, dim)
        full.mixer.queries.copy_(queries)
        block.mixer.queries.copy_(queries)

    embedding = torch.randn(2, dim)
    outputs = [torch.randn(2, dim) for _ in range(num_layers)]
    state = block.init_state(embedding)

    prior_outputs: list[torch.Tensor] = []
    for layer_index, output in enumerate(outputs):
        full_input, full_weights = full.layer_input_with_weights(
            embedding=embedding,
            prior_layer_outputs=prior_outputs,
            layer_index=layer_index,
        )
        block_input, block_weights = block.layer_input_with_weights(
            state=state,
            layer_index=layer_index,
        )
        assert torch.allclose(full_input, block_input, atol=1e-7, rtol=0.0)
        assert torch.allclose(full_weights, block_weights, atol=1e-7, rtol=0.0)

        prior_outputs.append(output)
        block.append_layer_output(state, output)

    full_final, full_final_weights = full.final_output_with_weights(
        embedding=embedding,
        layer_outputs=outputs,
    )
    block_final, block_final_weights = block.final_output_with_weights(state)

    assert torch.allclose(full_final, block_final, atol=1e-7, rtol=0.0)
    assert torch.allclose(full_final_weights, block_final_weights, atol=1e-7, rtol=0.0)
