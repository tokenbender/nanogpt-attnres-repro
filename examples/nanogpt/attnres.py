from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _stack_sources(source_values: Sequence[Tensor] | Tensor) -> Tensor:
    if isinstance(source_values, torch.Tensor):
        if source_values.ndim < 2:
            raise ValueError("source_values tensor must have shape [S, ..., D]")
        return source_values

    if len(source_values) == 0:
        raise ValueError("source_values must contain at least one tensor")

    reference_shape = source_values[0].shape
    for source in source_values[1:]:
        if source.shape != reference_shape:
            raise ValueError("all source tensors must share the same shape")

    return torch.stack(list(source_values), dim=0)


class ReferenceRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        compute_dtype = (
            x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        )
        rms = x.to(compute_dtype).pow(2).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(rms + self.eps).to(dtype=x.dtype)
        weight = self.weight.to(dtype=x.dtype)
        return x * scale * weight


class DepthSoftmaxMixer(nn.Module):
    def __init__(self, num_queries: int, dim: int, eps: float = 1e-8):
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.num_queries = num_queries
        self.dim = dim
        self.key_norms = nn.ModuleList(
            [ReferenceRMSNorm(dim=dim, eps=eps) for _ in range(num_queries)]
        )
        self.queries = nn.Parameter(torch.zeros(num_queries, dim))

    def _validate_query_index(self, query_index: int) -> None:
        if not 0 <= query_index < self.num_queries:
            raise IndexError(
                f"query_index={query_index} out of range for {self.num_queries} queries"
            )

    def _validate_source_tensor(self, source_tensor: Tensor) -> None:
        if source_tensor.ndim < 2:
            raise ValueError("source_tensor must have shape [S, ..., D]")
        if source_tensor.shape[-1] != self.dim:
            raise ValueError(
                f"expected trailing dim {self.dim}, received {source_tensor.shape[-1]}"
            )

    def mix_with_weights(
        self, query_index: int, source_values: Sequence[Tensor] | Tensor
    ) -> tuple[Tensor, Tensor]:
        self._validate_query_index(query_index)
        source_tensor = _stack_sources(source_values)
        self._validate_source_tensor(source_tensor)

        normed_sources = self.key_norms[query_index](source_tensor)
        query = self.queries[query_index].to(dtype=source_tensor.dtype)
        logits = torch.einsum("d,s...d->s...", query, normed_sources)
        weights = F.softmax(logits, dim=0)
        mixed = torch.einsum("s...,s...d->...d", weights, source_tensor)
        return mixed, weights

    def forward(
        self, query_index: int, source_values: Sequence[Tensor] | Tensor
    ) -> Tensor:
        mixed, _ = self.mix_with_weights(
            query_index=query_index, source_values=source_values
        )
        return mixed


class FullAttnResReference(nn.Module):
    def __init__(self, num_layers: int, dim: int, eps: float = 1e-8):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self.dim = dim
        self.mixer = DepthSoftmaxMixer(num_queries=num_layers + 1, dim=dim, eps=eps)

    def _sources_for_layer(
        self, embedding: Tensor, prior_layer_outputs: Sequence[Tensor], layer_index: int
    ) -> list[Tensor]:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index={layer_index} out of range for {self.num_layers} layers"
            )
        if len(prior_layer_outputs) != layer_index:
            raise ValueError(
                "Full AttnRes requires one prior layer output per completed logical layer"
            )
        return [embedding, *prior_layer_outputs]

    def layer_input_with_weights(
        self, embedding: Tensor, prior_layer_outputs: Sequence[Tensor], layer_index: int
    ) -> tuple[Tensor, Tensor]:
        sources = self._sources_for_layer(
            embedding=embedding,
            prior_layer_outputs=prior_layer_outputs,
            layer_index=layer_index,
        )
        return self.mixer.mix_with_weights(
            query_index=layer_index, source_values=sources
        )

    def layer_input(
        self, embedding: Tensor, prior_layer_outputs: Sequence[Tensor], layer_index: int
    ) -> Tensor:
        mixed, _ = self.layer_input_with_weights(
            embedding=embedding,
            prior_layer_outputs=prior_layer_outputs,
            layer_index=layer_index,
        )
        return mixed

    def final_output_with_weights(
        self, embedding: Tensor, layer_outputs: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor]:
        if len(layer_outputs) != self.num_layers:
            raise ValueError("final_output requires all logical layer outputs")
        sources = [embedding, *layer_outputs]
        return self.mixer.mix_with_weights(
            query_index=self.num_layers,
            source_values=sources,
        )

    def final_output(
        self, embedding: Tensor, layer_outputs: Sequence[Tensor]
    ) -> Tensor:
        mixed, _ = self.final_output_with_weights(
            embedding=embedding, layer_outputs=layer_outputs
        )
        return mixed


@dataclass
class BlockAttnResState:
    completed_blocks: list[Tensor]
    partial_block: Tensor | None
    completed_layers: int
    block_size: int

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if len(self.completed_blocks) == 0:
            raise ValueError("completed_blocks must include the embedding source")
        reference_shape = self.completed_blocks[0].shape
        for block in self.completed_blocks[1:]:
            if block.shape != reference_shape:
                raise ValueError("completed block shapes must match")
        if (
            self.partial_block is not None
            and self.partial_block.shape != reference_shape
        ):
            raise ValueError("partial_block must match completed block shape")
        if self.completed_layers < 0:
            raise ValueError("completed_layers cannot be negative")

    @classmethod
    def initialize(cls, embedding: Tensor, block_size: int) -> "BlockAttnResState":
        return cls(
            completed_blocks=[embedding],
            partial_block=None,
            completed_layers=0,
            block_size=block_size,
        )

    def current_sources(self) -> list[Tensor]:
        if self.partial_block is None:
            return list(self.completed_blocks)
        return [*self.completed_blocks, self.partial_block]

    def append_layer_output(self, layer_output: Tensor) -> None:
        reference_shape = self.completed_blocks[0].shape
        if layer_output.shape != reference_shape:
            raise ValueError("layer_output must match embedding shape")

        if self.partial_block is None:
            self.partial_block = layer_output
        else:
            self.partial_block = self.partial_block + layer_output

        self.completed_layers += 1

        if self.completed_layers % self.block_size == 0:
            self.completed_blocks.append(self.partial_block)
            self.partial_block = None

    def final_sources(self) -> list[Tensor]:
        sources = list(self.completed_blocks)
        if self.partial_block is not None:
            sources.append(self.partial_block)
        return sources


class BlockAttnResReference(nn.Module):
    def __init__(self, num_layers: int, dim: int, block_size: int, eps: float = 1e-8):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.num_layers = num_layers
        self.dim = dim
        self.block_size = block_size
        self.mixer = DepthSoftmaxMixer(num_queries=num_layers + 1, dim=dim, eps=eps)

    def init_state(self, embedding: Tensor) -> BlockAttnResState:
        if embedding.shape[-1] != self.dim:
            raise ValueError(
                f"expected embedding trailing dim {self.dim}, received {embedding.shape[-1]}"
            )
        return BlockAttnResState.initialize(
            embedding=embedding, block_size=self.block_size
        )

    def layer_input_with_weights(
        self, state: BlockAttnResState, layer_index: int | None = None
    ) -> tuple[Tensor, Tensor]:
        current_layer = state.completed_layers if layer_index is None else layer_index
        if current_layer != state.completed_layers:
            raise ValueError(
                "layer_index must equal state.completed_layers for exact bookkeeping"
            )
        if not 0 <= current_layer < self.num_layers:
            raise IndexError(
                f"layer_index={current_layer} out of range for {self.num_layers} layers"
            )
        return self.mixer.mix_with_weights(
            query_index=current_layer,
            source_values=state.current_sources(),
        )

    def layer_input(
        self, state: BlockAttnResState, layer_index: int | None = None
    ) -> Tensor:
        mixed, _ = self.layer_input_with_weights(state=state, layer_index=layer_index)
        return mixed

    def append_layer_output(
        self, state: BlockAttnResState, layer_output: Tensor
    ) -> None:
        state.append_layer_output(layer_output)

    def final_output_with_weights(
        self, state: BlockAttnResState
    ) -> tuple[Tensor, Tensor]:
        if state.completed_layers != self.num_layers:
            raise ValueError(
                "final_output requires all logical layer outputs to be appended"
            )
        return self.mixer.mix_with_weights(
            query_index=self.num_layers,
            source_values=state.final_sources(),
        )

    def final_output(self, state: BlockAttnResState) -> Tensor:
        mixed, _ = self.final_output_with_weights(state=state)
        return mixed


__all__ = [
    "BlockAttnResReference",
    "BlockAttnResState",
    "DepthSoftmaxMixer",
    "FullAttnResReference",
    "ReferenceRMSNorm",
]
