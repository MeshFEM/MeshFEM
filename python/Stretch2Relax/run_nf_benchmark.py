#!/usr/bin/env python3
"""Launch the Newton-flow surface-parameterization benchmark from the command line."""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import groupby
from pathlib import Path


def _configure_native_thread_environment() -> None:
    """Default non-TBB native thread pools to one before importing bindings."""

    for variable in (
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


_configure_native_thread_environment()

_SCRIPT_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = _SCRIPT_DIR.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from nf_benchmark_utils import (  # noqa: E402
    BaseInitializationCache,
    BenchmarkConfig,
    CsvIterationWriter,
    InitializationContext,
    InitialOptimizerRegistry,
    InitializerRegistry,
    MethodRegistry,
    RunSpec,
    build_common_problem_bundle,
    build_experiment_matrix,
    discover_models,
    expand_method_specs,
    format_method_variant,
    load_normalized_mesh,
    native_thread_environment,
    parse_initial_optimizer_specs,
    parse_initializer_specs,
    register_builtin_initial_optimizers,
    register_builtin_initializers,
    register_builtin_methods,
    run_complete_method,
    set_thread_limit,
    timed_initialize_uv,
    validate_config,
)


class _BenchmarkHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show argument defaults while preserving multiline help examples."""


def build_parser(
    method_registry: MethodRegistry,
    initializer_registry: InitializerRegistry,
    initial_optimizer_registry: InitialOptimizerRegistry,
) -> argparse.ArgumentParser:
    """Build the common CLI and allow registered adapters to extend it."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Newton, Newton-flow extrapolators, and rotation-strain "
            "extrapolation on pre-cut surface meshes."
        ),
        epilog="""Examples:
  Run Newton and Taylor2 with two initializers using otherwise default settings:
    python run_nf_benchmark.py \\
      --models-dir /path/to/precut-models \\
      --output-csv exp_results/benchmark.csv \\
      --methods Newton Taylor2 \\
      --initializers tutte energy_minimal \\
      --constant-speed false

  Run a larger matrix over threads, repeats, and initial-optimization settings:
    python run_nf_benchmark.py \\
      --models-dir /path/to/precut-models \\
      --output-csv exp_results/benchmark.csv \\
      --methods Newton Taylor3 Pade14 RS \\
      --initializers tutte energy_minimal grad_minimal \\
      --constant-speed both \\
      --threads 1 2 4 8 16 \\
      --repeat 3 \\
      --max-iters 100 \\
      --initial-optimizer Newton \\
      --initial-optimization-iters 0 5 10

  Preview that larger experiment matrix without running it or writing the CSV:
    python run_nf_benchmark.py \\
      --models-dir /path/to/precut-models \\
      --output-csv exp_results/benchmark.csv \\
      --methods Newton Taylor3 Pade14 RS \\
      --initializers tutte energy_minimal grad_minimal \\
      --constant-speed both \\
      --threads 1 2 4 8 16 \\
      --repeat 3 \\
      --initial-optimizer Newton \\
      --initial-optimization-iters 0 5 10 \\
      --dry-run
""",
        formatter_class=_BenchmarkHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Directory containing immediate OBJ, OFF, or MSH model files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="CSV file to create or append after every completed iteration.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        metavar="METHOD",
        help=f"Method tokens. Registered forms: {method_registry.help_summary()}.",
    )
    parser.add_argument(
        "--initializers",
        nargs="+",
        default=["tutte"],
        metavar="INITIALIZER",
        help=(
            "Initializer presets. Registered presets: "
            f"{initializer_registry.help_summary()}."
        ),
    )
    parser.add_argument(
        "--constant-speed",
        type=str.lower,
        choices=("true", "false", "both"),
        default=None,
        help="Newton-flow constant-speed variant selection; required for Taylor/Pade.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[8],
        metavar="N",
        help="Maximum TBB thread counts to benchmark.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of complete repetitions for every configuration.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=100,
        help="Total iteration budget for each complete method run.",
    )
    parser.add_argument(
        "--initial-optimizer",
        nargs="+",
        default=["Newton"],
        metavar="OPTIMIZER",
        help=(
            "Phase-1 optimizers used for positive iteration caps. Registered names: "
            f"{initial_optimizer_registry.help_summary()}."
        ),
    )
    parser.add_argument(
        "--initial-optimization-iters",
        type=int,
        nargs="+",
        default=[10],
        metavar="N",
        help=(
            "One or more Phase-1 iteration limits included within --max-iters "
            "for eligible methods; zero creates one collapsed N/A core-only variant."
        ),
    )
    parser.add_argument(
        "--initial-newton-iters",
        dest="legacy_initial_newton_iters",
        type=int,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved experiment matrix without running or writing CSV output.",
    )

    for adapter in (
        *method_registry.adapters(),
        *initializer_registry.adapters(),
        *initial_optimizer_registry.adapters(),
    ):
        if adapter.add_cli_arguments is not None:
            adapter.add_cli_arguments(parser)
    return parser


def make_config(
    args: argparse.Namespace,
    initializer_specs: list,
    initial_optimizer_specs: list,
) -> BenchmarkConfig:
    """Convert parsed arguments into an immutable benchmark configuration."""

    return BenchmarkConfig(
        models_dir=args.models_dir.expanduser().resolve(),
        output_csv=args.output_csv.expanduser().resolve(),
        requested_method_tokens=tuple(args.methods),
        constant_speed_mode=args.constant_speed,
        thread_counts=tuple(args.threads),
        repeat_count=args.repeat,
        max_iters=args.max_iters,
        initial_optimizer_specs=tuple(initial_optimizer_specs),
        initial_optimization_iters=tuple(args.initial_optimization_iters),
        initializer_specs=tuple(initializer_specs),
    )


def print_configuration(
    config: BenchmarkConfig,
    models: list[Path],
    methods: list,
    total_runs: int,
) -> None:
    """Print the resolved experiment matrix and thread-pool provenance."""

    print("Newton-flow benchmark configuration")
    print(f"  models_dir: {config.models_dir}")
    print(f"  models ({len(models)}): {', '.join(path.name for path in models)}")
    print(
        "  initializers: "
        + ", ".join(spec.canonical_name for spec in config.initializer_specs)
    )
    print("  methods: " + ", ".join(format_method_variant(spec) for spec in methods))
    print(
        "  initial_optimizers: "
        + ", ".join(spec.canonical_name for spec in config.initial_optimizer_specs)
    )
    print(f"  threads: {list(config.thread_counts)}")
    print(f"  repeat: {config.repeat_count}")
    print(f"  max_iters: {config.max_iters}")
    print(
        "  initial_optimization_iters: "
        f"{list(config.initial_optimization_iters)}"
    )
    print(f"  output_csv: {config.output_csv}")
    print(f"  native_thread_environment: {native_thread_environment()}")
    print(f"  complete_runs: {total_runs}")


def print_experiment_matrix(matrix: list[RunSpec]) -> None:
    """Print the resolved dry-run matrix as an aligned terminal table."""

    headers = (
        "Run",
        "ID",
        "Model",
        "Initializer",
        "Method",
        "Constant speed",
        "Threads",
        "Repeat",
        "Initial optimizer",
        "Initial iters",
        "Max iters",
    )
    rows = [
        (
            str(index),
            str(run_spec.run_id),
            run_spec.model_path.name,
            run_spec.initializer_spec.canonical_name,
            run_spec.method_spec.display_name,
            (
                "NA"
                if run_spec.method_spec.constant_speed is None
                else str(run_spec.method_spec.constant_speed)
            ),
            str(run_spec.thread_num),
            str(run_spec.repeat_index),
            run_spec.initial_optimizer_name,
            str(run_spec.initial_optimization_iters),
            str(run_spec.max_iters),
        )
        for index, run_spec in enumerate(matrix, start=1)
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        """Align one matrix row to the computed column widths."""

        return "  ".join(value.ljust(width) for value, width in zip(row, widths))

    print(f"\nExperiment matrix ({len(matrix)} complete runs)")
    print(format_row(headers))
    print(format_row(tuple("-" * width for width in widths)))
    for row in rows:
        print(format_row(row))


def run_benchmark(
    config: BenchmarkConfig,
    matrix: list[RunSpec],
    method_registry: MethodRegistry,
    initializer_registry: InitializerRegistry,
    initial_optimizer_registry: InitialOptimizerRegistry,
) -> int:
    """Execute the same explicit matrix displayed by --dry-run."""

    benchmark_start = time.perf_counter()
    total_runs = len(matrix)
    run_positions = {run_spec.run_id: index for index, run_spec in enumerate(matrix, 1)}
    attempted_runs = 0
    failed_configurations = 0

    with CsvIterationWriter(config.output_csv) as writer:
        for model_path, model_group in groupby(matrix, key=lambda spec: spec.model_path):
            model_runs = list(model_group)
            try:
                normalized_mesh = load_normalized_mesh(model_path)
            except Exception as exc:
                failed_configurations += len(model_runs)
                print(
                    f"[model failure] {model_path.name}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue

            initialization_context = InitializationContext(
                normalized_mesh=normalized_mesh,
                model_path=model_path,
                base_cache=BaseInitializationCache(),
            )

            initializer_groups = groupby(
                model_runs,
                key=lambda spec: spec.initializer_spec.canonical_name,
            )
            for _, initializer_group in initializer_groups:
                initializer_runs = list(initializer_group)
                initializer_spec = initializer_runs[0].initializer_spec
                set_thread_limit(1)
                try:
                    initialization, initialization_elapsed_ns = timed_initialize_uv(
                        initialization_context,
                        initializer_spec,
                        initializer_registry,
                    )
                except Exception as exc:
                    failed_configurations += len(initializer_runs)
                    print(
                        f"[initializer failure] {model_path.name} / "
                        f"{initializer_spec.canonical_name}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                scale_factor = initialization.metadata.get("scale_factor", "NA")
                print(
                    f"[initialized] {model_path.name} / {initialization.name} "
                    f"(base={initialization.base_initializer}, scale={scale_factor}, "
                    f"time={initialization_elapsed_ns / 1_000_000_000.0:.6f}s)"
                )

                for run_spec in initializer_runs:
                    run_index = run_positions[run_spec.run_id]
                    attempted_runs += 1
                    set_thread_limit(run_spec.thread_num)
                    variant_name = format_method_variant(run_spec.method_spec)
                    print(
                        f"[run {run_index}/{total_runs}] id={run_spec.run_id} "
                        f"model={run_spec.model_name} "
                        f"initializer={run_spec.initializer_spec.canonical_name} "
                        f"method={variant_name} "
                        f"initial_optimizer={run_spec.initial_optimizer_name} "
                        f"initial_optimization_iters="
                        f"{run_spec.initial_optimization_iters} "
                        f"threads={run_spec.thread_num} "
                        f"repeat={run_spec.repeat_index}"
                    )

                    try:
                        bundle = build_common_problem_bundle(
                            normalized_mesh,
                            initialization.uv.copy(),
                            config.max_iters,
                        )
                        outcome = run_complete_method(
                            bundle,
                            run_spec,
                            writer,
                            method_registry,
                            initial_optimizer_registry,
                            initialization_elapsed_ns,
                        )
                    except Exception as exc:
                        failed_configurations += 1
                        print(
                            f"[run failure] id={run_spec.run_id} "
                            f"model={run_spec.model_name} "
                            f"initializer={run_spec.initializer_spec.canonical_name} "
                            f"method={variant_name} "
                            f"initial_optimizer={run_spec.initial_optimizer_name} "
                            f"initial_optimization_iters="
                            f"{run_spec.initial_optimization_iters} "
                            f"threads={run_spec.thread_num} "
                            f"repeat={run_spec.repeat_index}: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                        continue

                    status = "failed" if outcome.failed else (
                        "converged" if outcome.converged else "stopped"
                    )
                    print(
                        f"[done {run_index}/{total_runs}] status={status} "
                        f"iterations={outcome.completed_iterations} "
                        f"reason={outcome.termination_reason}"
                    )
                    if outcome.failed:
                        failed_configurations += 1

    global_elapsed_seconds = time.perf_counter() - benchmark_start
    print(
        f"Benchmark finished: attempted={attempted_runs}, expected={total_runs}, "
        f"failed_or_skipped={failed_configurations}, "
        f"global_elapsed_seconds={global_elapsed_seconds:.6f}, "
        f"output={config.output_csv}"
    )
    return 1 if failed_configurations else 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, validate the matrix, and launch the benchmark."""

    method_registry = register_builtin_methods()
    initializer_registry = register_builtin_initializers()
    initial_optimizer_registry = register_builtin_initial_optimizers()
    parser = build_parser(
        method_registry,
        initializer_registry,
        initial_optimizer_registry,
    )
    args = parser.parse_args(argv)
    if args.legacy_initial_newton_iters is not None:
        parser.error(
            "--initial-newton-iters was renamed; use "
            "--initial-optimization-iters"
        )

    try:
        initializer_specs = parse_initializer_specs(args, initializer_registry)
        initial_optimizer_specs = parse_initial_optimizer_specs(
            args,
            initial_optimizer_registry,
        )
        config = make_config(args, initializer_specs, initial_optimizer_specs)
        validate_config(config)
        methods = expand_method_specs(
            config.requested_method_tokens,
            config.constant_speed_mode,
            method_registry,
        )
        models = discover_models(config.models_dir)
        experiment_matrix = build_experiment_matrix(
            config,
            models,
            methods,
            method_registry,
        )
        total_runs = len(experiment_matrix)
    except ValueError as exc:
        parser.error(str(exc))

    if config.constant_speed_mode is not None and not any(
        method_registry.get(spec.adapter_key).uses_constant_speed for spec in methods
    ):
        print("Notice: --constant-speed is ignored because no selected method uses it.")
    if not any(
        method_registry.get(spec.adapter_key).uses_initial_optimization
        for spec in methods
    ):
        print(
            "Notice: --initial-optimizer and --initial-optimization-iters are "
            "ignored because no selected method uses Phase 1."
        )

    print_configuration(config, models, methods, total_runs)
    if args.dry_run:
        print_experiment_matrix(experiment_matrix)
        print("\nDry run complete: no meshes were loaded and no CSV output was written.")
        return 0

    return run_benchmark(
        config,
        experiment_matrix,
        method_registry,
        initializer_registry,
        initial_optimizer_registry,
    )


if __name__ == "__main__":
    raise SystemExit(main())
