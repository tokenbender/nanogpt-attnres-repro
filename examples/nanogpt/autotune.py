from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from ast import literal_eval
from datetime import datetime
from pathlib import Path


DEFAULT_CONFIG = {
    "batch_size": 64,
    "block_size": 256,
    "gradient_accumulation_steps": 1,
    "target_tokens_per_iter": None,
    "target_tokens": None,
}


def parse_override(text: str) -> tuple[str, object]:
    key, value = text.split("=", 1)
    try:
        parsed = literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    return key, parsed


def load_config_values(config_path: Path, overrides: list[str]) -> dict[str, object]:
    values = dict(DEFAULT_CONFIG)
    namespace: dict[str, object] = {}
    exec(config_path.read_text(encoding="utf-8"), namespace)
    for key, value in namespace.items():
        if not key.startswith("__"):
            values[key] = value
    for override in overrides:
        key, value = parse_override(override)
        values[key] = value
    return values


def derive_target_tokens_per_iter(config: dict[str, object]) -> int:
    target = config.get("target_tokens_per_iter")
    if target is not None:
        return int(target)
    return (
        int(config["batch_size"])
        * int(config["block_size"])
        * int(config["gradient_accumulation_steps"])
    )


def derive_total_accumulation(
    *, batch_size: int, block_size: int, target_tokens_per_iter: int
) -> int | None:
    micro_step_tokens = batch_size * block_size
    if micro_step_tokens <= 0:
        return None
    if target_tokens_per_iter % micro_step_tokens != 0:
        return None
    return target_tokens_per_iter // micro_step_tokens


def is_legal_batch_size(
    *, batch_size: int, block_size: int, target_tokens_per_iter: int, world_size: int
) -> bool:
    total_accum = derive_total_accumulation(
        batch_size=batch_size,
        block_size=block_size,
        target_tokens_per_iter=target_tokens_per_iter,
    )
    if total_accum is None:
        return False
    return total_accum > 0 and total_accum % world_size == 0


def default_batch_sizes(
    *,
    base_batch_size: int,
    block_size: int,
    target_tokens_per_iter: int,
    world_size: int,
) -> list[int]:
    max_batch_size = target_tokens_per_iter // block_size
    candidates: set[int] = set()
    candidate = 1
    while candidate <= max_batch_size:
        if is_legal_batch_size(
            batch_size=candidate,
            block_size=block_size,
            target_tokens_per_iter=target_tokens_per_iter,
            world_size=world_size,
        ):
            candidates.add(candidate)
        candidate *= 2

    if is_legal_batch_size(
        batch_size=base_batch_size,
        block_size=block_size,
        target_tokens_per_iter=target_tokens_per_iter,
        world_size=world_size,
    ):
        candidates.add(base_batch_size)

    return sorted(candidates)


def build_train_command(
    *,
    config_path: Path,
    out_dir: Path,
    nproc_per_node: int,
    batch_size: int,
    compile_model: bool,
    benchmark_steps: int,
    target_tokens_per_iter: int,
    data_dir: str | None,
    extra_overrides: list[str],
) -> list[str]:
    if nproc_per_node > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
            "train.py",
            str(config_path),
        ]
    else:
        cmd = [sys.executable, "train.py", str(config_path)]

    overrides = [
        f"out_dir={repr(str(out_dir))}",
        "wandb_log=False",
        f"batch_size={batch_size}",
        f"compile_model={compile_model}",
        f"target_tokens_per_iter={target_tokens_per_iter}",
        "target_tokens=None",
        "lock_lr_decay_to_max_iters=False",
        f"max_iters={benchmark_steps}",
        f"lr_decay_iters={benchmark_steps}",
        "eval_interval=1000000",
        "eval_iters=1",
        "log_interval=1",
    ]
    if data_dir is not None:
        overrides.append(f"data_dir={repr(data_dir)}")
    overrides.extend(extra_overrides)
    return [*cmd, *overrides]


def load_json(path: Path) -> dict[str, object]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def run_candidate(
    *,
    nanogpt_dir: Path,
    config_path: Path,
    result_dir: Path,
    nproc_per_node: int,
    batch_size: int,
    compile_model: bool,
    benchmark_steps: int,
    target_tokens_per_iter: int,
    data_dir: str | None,
    extra_overrides: list[str],
) -> dict[str, object]:
    result_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = result_dir / "stdout.log"
    cmd = build_train_command(
        config_path=config_path,
        out_dir=result_dir,
        nproc_per_node=nproc_per_node,
        batch_size=batch_size,
        compile_model=compile_model,
        benchmark_steps=benchmark_steps,
        target_tokens_per_iter=target_tokens_per_iter,
        data_dir=data_dir,
        extra_overrides=extra_overrides,
    )

    started = time.time()
    with stdout_path.open("wb") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(nanogpt_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
    elapsed = time.time() - started

    summary_path = result_dir / "summary.json"
    config_effective_path = result_dir / "config_effective.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    config_effective = (
        load_json(config_effective_path) if config_effective_path.exists() else {}
    )
    last_train_log = summary.get("last_train_log") or {}

    return {
        "status": "ok" if proc.returncode == 0 and summary.get("ok") else "failed",
        "returncode": proc.returncode,
        "batch_size": batch_size,
        "compile_model": compile_model,
        "world_size": nproc_per_node,
        "elapsed_s": elapsed,
        "tokens_per_iter": summary.get("tokens_per_iter"),
        "iter_num": summary.get("iter_num"),
        "tok_per_sec": last_train_log.get("tok_per_sec"),
        "iter_time_ms": last_train_log.get("iter_time_ms"),
        "max_mem_allocated_mb": last_train_log.get("max_mem_allocated_mb"),
        "max_mem_reserved_mb": last_train_log.get("max_mem_reserved_mb"),
        "gradient_accumulation_steps_total": config_effective.get(
            "gradient_accumulation_steps_total"
        ),
        "gradient_accumulation_steps_per_rank": config_effective.get(
            "gradient_accumulation_steps_per_rank"
        ),
        "out_dir": str(result_dir),
        "stdout_log": str(stdout_path),
        "error": summary.get("error", "") if summary else "",
    }


def write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "status",
        "batch_size",
        "compile_model",
        "world_size",
        "gradient_accumulation_steps_total",
        "gradient_accumulation_steps_per_rank",
        "tokens_per_iter",
        "iter_num",
        "tok_per_sec",
        "iter_time_ms",
        "max_mem_allocated_mb",
        "max_mem_reserved_mb",
        "elapsed_s",
        "returncode",
        "out_dir",
        "stdout_log",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_best(rows: list[dict[str, object]]) -> dict[str, object] | None:
    ok_rows = [
        row for row in rows if row["status"] == "ok" and row["tok_per_sec"] is not None
    ]
    if not ok_rows:
        return None
    return max(ok_rows, key=lambda row: float(row["tok_per_sec"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autotune fair nanoGPT batch settings")
    parser.add_argument(
        "config", help="Path to train config, e.g. config/train_fineweb10B.py"
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of local processes / GPUs to benchmark with",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Explicit per-device batch sizes to try",
    )
    parser.add_argument(
        "--compile-candidates",
        nargs="*",
        choices=["false", "true"],
        default=["false"],
        help="Whether to benchmark torch.compile candidates as well",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=6,
        help="Number of short training iterations per candidate",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional dataset directory override",
    )
    parser.add_argument(
        "--out-root",
        default="../../experiments/autotune",
        help="Directory for autotune outputs, relative to examples/nanogpt",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional train.py override, repeatable (key=value)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nanogpt_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (nanogpt_dir / config_path).resolve()

    loaded = load_config_values(config_path, args.override)
    block_size = int(loaded["block_size"])
    base_batch_size = int(loaded["batch_size"])
    target_tokens_per_iter = derive_target_tokens_per_iter(loaded)

    if args.batch_sizes is None:
        batch_sizes = default_batch_sizes(
            base_batch_size=base_batch_size,
            block_size=block_size,
            target_tokens_per_iter=target_tokens_per_iter,
            world_size=args.nproc_per_node,
        )
    else:
        batch_sizes = [
            bs
            for bs in sorted(set(args.batch_sizes))
            if is_legal_batch_size(
                batch_size=bs,
                block_size=block_size,
                target_tokens_per_iter=target_tokens_per_iter,
                world_size=args.nproc_per_node,
            )
        ]

    if not batch_sizes:
        raise SystemExit(
            "No legal batch sizes found for this semantic target and world size"
        )

    compile_candidates = [candidate == "true" for candidate in args.compile_candidates]

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_root = (
        nanogpt_dir / args.out_root / f"{config_path.stem}-{timestamp}"
    ).resolve()
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Autotuning {config_path.name}")
    print(f"  world_size={args.nproc_per_node}")
    print(f"  target_tokens_per_iter={target_tokens_per_iter}")
    print(f"  candidate_batch_sizes={batch_sizes}")
    print(f"  compile_candidates={compile_candidates}")
    print()

    rows: list[dict[str, object]] = []
    for compile_model in compile_candidates:
        for batch_size in batch_sizes:
            result_dir = run_root / f"bs{batch_size}-compile{int(compile_model)}"
            print(f"Running batch_size={batch_size}, compile_model={compile_model}")
            row = run_candidate(
                nanogpt_dir=nanogpt_dir,
                config_path=config_path,
                result_dir=result_dir,
                nproc_per_node=args.nproc_per_node,
                batch_size=batch_size,
                compile_model=compile_model,
                benchmark_steps=args.benchmark_steps,
                target_tokens_per_iter=target_tokens_per_iter,
                data_dir=args.data_dir,
                extra_overrides=args.override,
            )
            rows.append(row)
            print(
                f"  status={row['status']} tok/s={row['tok_per_sec']} "
                f"grad_accum_total={row['gradient_accumulation_steps_total']}"
            )

    csv_path = run_root / "results.csv"
    write_results_csv(csv_path, rows)
    best = choose_best(rows)

    print()
    print(f"Wrote results to {csv_path}")
    if best is None:
        print("No successful candidates found")
        return 1

    print("Best candidate:")
    print(f"  batch_size={best['batch_size']}")
    print(f"  compile_model={best['compile_model']}")
    print(f"  tok/s={best['tok_per_sec']}")
    print(
        "  recommended overrides: "
        f"batch_size={best['batch_size']} compile_model={best['compile_model']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
