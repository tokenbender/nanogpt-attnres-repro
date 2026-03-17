# nanoGPT Attention Residuals Repro

Standalone, correctness-first reproduction workspace for comparing four residual mechanisms in nanoGPT-scale training runs:

- baseline residual
- mHC (Manifold-Constrained Hyper-Connections)
- Full Attention Residuals
- Block Attention Residuals

This repo vendors the minimal nanoGPT + mHC experiment harness from `tokenbender/mHC-manifold-constrained-hyper-connections` and uses it as the starting point for a faithful AttnRes reproduction.

## Goals

- keep the training harness compact, easy to modify, and multi-GPU friendly
- implement each method directly from the paper equations
- require mathematical and integration tests before trusting any experiment result
- reproduce the mechanism-level claims of the AttnRes paper at nanoGPT scale

## Scope

This repo is for dense nanoGPT-scale mechanism experiments, not a full reproduction of Moonshot's Kimi Linear stack.

Expected targets:

- validation loss comparisons at equal token budget
- baseline vs mHC vs Full AttnRes vs Block AttnRes
- output-norm and gradient-norm diagnostics across depth
- depth-mixing / attention-over-depth visualizations
- block-size sweeps for Block AttnRes

Non-goals for the first phase:

- MoE / Kimi Linear reproduction
- pipeline-parallel caching from the AttnRes systems section
- large downstream benchmark suites from the paper

## Current Status

Active implementation workspace.

- vendored nanoGPT training harness
- vendored `hyper_connections` package for mHC experiments
- run-contract and mHC invariant tests copied in
- Full AttnRes reference and training-path integration are in progress

Legacy `HC` and `vRes` codepaths are currently vendored only because they are present in the inherited nanoGPT harness. They are not part of the primary comparison matrix for this repo.

See `docs/FEATURE_ISSUES.md` for the first feature issues to build.

## Paper References

- Attention Residuals: <https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf>
- Hyper-Connections: <https://arxiv.org/abs/2409.19606>
- mHC: <https://arxiv.org/abs/2512.24880>

## Repository Layout

- `examples/nanogpt/` - training entrypoint, configs, baseline model harness
- `hyper_connections/` - vendored residual-stream implementation used for mHC
- `tests/` - invariant and run-contract tests
- `docs/` - roadmap, issue seeds, paper-locked constraints

## Correctness Policy

Correctness policy:

- no approximate AttnRes implementation gets merged as if it were faithful
- every optimized residual mixer needs a small reference implementation
- degenerate and boundary cases must be tested before long runs are trusted
- experiment plots are only meaningful after invariants and integration tests pass

## Budgeting Strategy

- FineWeb comparison configs lock experiment semantics with `target_tokens_per_iter` and `target_tokens`
- tune `batch_size` for the available hardware, and let the trainer derive accumulation to preserve the semantic batch
- prefer higher real microbatch size and lower accumulation on larger GPUs, as long as the semantic target stays fixed

## Autotuning

- use `examples/nanogpt/autotune.py` to benchmark legal per-device batch sizes for a given config while keeping the semantic batch fixed
- example single-node run:

```bash
cd examples/nanogpt
python autotune.py config/train_fineweb10B.py --nproc-per-node 1 --data-dir /path/to/fineweb10B
```

- the autotuner writes benchmark artifacts and a `results.csv` under `experiments/autotune/`

## Local Data

The vendored train harness currently expects FineWeb-style binary shards.

- Place shards under `examples/nanogpt/data/fineweb10B/` or pass `data_dir=...` at launch time.

- The current downloader script lives at `examples/nanogpt/data/fineweb10B/download.py`.

## First Build Sequence

1. baseline + mHC correctness in the standalone repo
2. Full AttnRes reference implementation
3. Full AttnRes training-path integration
4. Block AttnRes implementation with paper-faithful block bookkeeping
5. diagnostics and multi-GPU comparison runs
