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

- vendored nanoGPT training harness
- vendored `hyper_connections` package for mHC experiments
- run-contract, invariant, and comparison-config tests are in place
- Full AttnRes and Block AttnRes are integrated into the training path
- the primary comparison matrix is the shared 48-layer / ~1B-token family in `experiments/comparison_matrix.csv`
- `run_matrix_48l.sh` and `run_matrix_48l_all.sh` launch the current canonical runs

Legacy `HC` and `vRes` codepaths are currently vendored only because they are present in the inherited nanoGPT harness. They are not part of the primary comparison matrix for this repo.

See `experiments/comparison_matrix.csv` for the current run plan and `docs/FEATURE_ISSUES.md` for the original implementation issue history.

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
python autotune.py config/train_fineweb10B_48l.py --nproc-per-node 1 --data-dir /path/to/fineweb10B
```

- the autotuner writes benchmark artifacts and a `results.csv` under `experiments/autotune/`

## Matrix Launcher

- `examples/nanogpt/run_matrix_48l.sh` launches the shared 48-layer comparison family with one CLI argument
- `examples/nanogpt/run_matrix_48l_all.sh` launches the canonical four-run sequence sequentially
- the default sequential order is `attnres-full`, `attnres-block`, `mhc`, `baseline`
- edit the variables at the top of those files to change the W&B project, group, world size, batch size, shared overrides, or default sequence
- example usage:

```bash
cd examples/nanogpt
./run_matrix_48l.sh baseline
./run_matrix_48l.sh mhc
./run_matrix_48l.sh attnres-full
./run_matrix_48l.sh attnres-block
./run_matrix_48l.sh attnres-full --dry-run
./run_matrix_48l_all.sh
```

## Local Data

The vendored train harness currently expects FineWeb-style binary shards.

- Place shards under `examples/nanogpt/data/fineweb10B/` or pass `data_dir=...` at launch time.
- The current downloader script lives at `examples/nanogpt/data/fineweb10B/download.py`.

```bash
cd examples/nanogpt/data/fineweb10B
python download.py 103
```

## Current Experiment Flow

1. download FineWeb shards with `examples/nanogpt/data/fineweb10B/download.py`
2. check or edit the canonical run settings in `examples/nanogpt/run_matrix_48l.sh`
3. preview commands with `./run_matrix_48l.sh <algo> --dry-run` or `./run_matrix_48l_all.sh --dry-run`
4. run the canonical 48-layer matrix one-by-one with `./run_matrix_48l.sh <algo>` or sequentially with `./run_matrix_48l_all.sh`
5. record results back into `experiments/comparison_matrix.csv`
