# nanoGPT Attention Residuals Repro

Standalone, correctness-first reproduction workspace for comparing four residual mechanisms in nanoGPT-scale training runs:

- baseline residual
- mHC (Manifold-Constrained Hyper-Connections)
- Full Attention Residuals
- Block Attention Residuals

This repo vendors the minimal nanoGPT + mHC experiment harness from `tokenbender/mHC-manifold-constrained-hyper-connections` and uses it as the starting point for a faithful AttnRes reproduction.

## Goals

- keep the training harness small, hackable, and multi-GPU friendly
- implement each method against paper equations, not hand-wavy approximations
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

Initial scaffold.

- vendored nanoGPT training harness
- vendored `hyper_connections` package for mHC baseline parity work
- run-contract and mHC invariant tests copied in
- AttnRes implementation still to be added

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

The bar in this repo is simple:

- no approximate AttnRes implementation gets merged as if it were faithful
- every optimized residual mixer needs a tiny reference implementation
- degenerate and boundary cases must be tested before long runs are trusted
- experiment plots are only meaningful after invariants and parity tests pass

## Local Data

The vendored train harness currently expects FineWeb-style binary shards.

The existing local source repo already contains a usable path for smoke runs:

- `/Users/tokenbender/Documents/mHC-manifold-constrained-hyper-connections/examples/nanogpt/data/fineweb10B/`

## First Build Sequence

1. baseline + mHC parity in the standalone repo
2. Full AttnRes reference implementation
3. Full AttnRes training-path integration
4. Block AttnRes implementation with paper-faithful block bookkeeping
5. diagnostics and multi-GPU comparison runs
