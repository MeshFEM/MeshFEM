"""Utilities for the Newton-flow large-scale parameterization benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from uuid import UUID, uuid4

import numpy as np


# Make the module importable from any working directory. Environment variables
# controlling native thread pools must be set by the launcher before this file is imported.
_MODULE_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = _MODULE_DIR.parent
_PROJECT_DIR = _PYTHON_DIR.parent
for _dependency_path in (
    _PYTHON_DIR,
    _PYTHON_DIR / "curved_linesearch",
    _PROJECT_DIR / "3rdparty" / "MeshFEM" / "python",
):
    _path_string = str(_dependency_path)
    if _path_string not in sys.path:
        sys.path.insert(0, _path_string)

import MeshFEM  # noqa: F401,E402 - initializes MeshFEM's Python bindings
import benchmark  # noqa: E402
import continuation_parametrization  # noqa: E402
import fast_newton_flow  # noqa: E402
import flip_avoiding_step_length  # noqa: E402
import mesh  # noqa: E402
import mesh_energy  # noqa: E402
import parallelism  # noqa: E402
import param_utils  # noqa: E402
import parametrization  # noqa: E402
import py_newton_optimizer  # noqa: E402
import rotation_strain_extrapolation  # noqa: E402
from Benchmark import helper_funcs  # noqa: E402

import extra_utils  # noqa: E402
import initial_utils  # noqa: E402
import opt_utils  # noqa: E402


GRADIENT_TOLERANCE = 2e-8
SUPPORTED_MODEL_EXTENSIONS = frozenset({".obj", ".off", ".msh"})
SCALE_INITIALIZER_NAMES = (
    "energy_minimal",
    "grad_minimal",
    "full_tension",
    "psd",
    "bulk_tension",
)
CSV_COLUMNS = (
    "ID",
    "model_name",
    "method",
    "initializer",
    "initial_optimizer",
    "initial_optimization_iters",
    "initial_optimization_grad_tol",
    "repeat_index",
    "constant_speed_indicator",
    "thread_num",
    "Phase",
    "iteration_index",
    "energy",
    "gradient_norm",
    "time_spent_within_this_iteration",
    "current_timestamp",
)
RUN_STATUS_COLUMNS = (
    "ID",
    "model_name",
    "method",
    "initializer",
    "initial_optimizer",
    "initial_optimization_iters",
    "initial_optimization_grad_tol",
    "repeat_index",
    "constant_speed_indicator",
    "thread_num",
    "max_iters",
    "status",
    "termination_reason",
    "failure_scope",
    "completed_iterations",
    "last_recorded_phase",
    "last_recorded_iteration",
    "exception_type",
    "exception_message",
    "finished_timestamp",
)
RUN_STATUS_VALUES = frozenset({"converged", "completed", "failed", "skipped"})
FAILURE_SCOPE_VALUES = frozenset({"NA", "model_load", "initializer", "run_execution"})


class BenchmarkPhase(IntEnum):
    """Identify UV initialization, initial optimization, and core optimization."""

    UV_INITIALIZATION = 0
    INITIAL_OPTIMIZATION = 1
    CORE_METHOD = 2


@dataclass(frozen=True)
class InitializerSpec:
    """Describe one normalized initializer preset requested on the command line."""

    adapter_key: str
    canonical_name: str
    options: Mapping[str, Any] = field(default_factory=dict)
    random_seed: Optional[int] = None


@dataclass(frozen=True)
class MethodSpec:
    """Describe one concrete method variant after token and speed expansion."""

    adapter_key: str
    display_name: str
    options: Mapping[str, Any] = field(default_factory=dict)
    constant_speed: Optional[bool] = None


@dataclass(frozen=True)
class InitialOptimizerSpec:
    """Describe one normalized initial optimizer selected on the command line."""

    adapter_key: str
    canonical_name: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Hold validated launcher configuration shared by all benchmark runs."""

    models_dir: Path
    output_csv: Path
    requested_method_tokens: tuple[str, ...]
    constant_speed_mode: Optional[str]
    thread_counts: tuple[int, ...]
    repeat_count: int
    max_iters: int
    initial_optimizer_specs: tuple[InitialOptimizerSpec, ...]
    initial_optimization_iters: tuple[int, ...]
    initializer_specs: tuple[InitializerSpec, ...]
    initial_only: bool = False
    initial_optimization_grad_tols: tuple[float, ...] = (GRADIENT_TOLERANCE,)
    models_dir_recursive: bool = False


@dataclass(frozen=True)
class RunSpec:
    """Identify one complete model/initializer/method/thread/repeat run."""

    run_id: UUID
    model_path: Path
    model_name: str
    initializer_spec: InitializerSpec
    method_spec: Optional[MethodSpec]
    initial_optimizer_spec: Optional[InitialOptimizerSpec]
    initial_optimization_iters: int
    thread_num: int
    repeat_index: int
    max_iters: int
    initial_optimization_grad_tol: Optional[float] = GRADIENT_TOLERANCE

    @property
    def method_name(self) -> str:
        """Return the selected core method or the initial-only sentinel."""

        if self.method_spec is None:
            return "N/A"
        return self.method_spec.display_name

    @property
    def initial_optimizer_name(self) -> str:
        """Return the canonical optimizer name or the disabled-stage sentinel."""

        if self.initial_optimizer_spec is None:
            return "N/A"
        return self.initial_optimizer_spec.canonical_name


@dataclass(frozen=True)
class BaseInitialization:
    """Store an unprocessed base embedding and its diagnostics."""

    name: str
    uv: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitializationResult:
    """Store finalized UV values and provenance for one initializer preset."""

    name: str
    uv: np.ndarray
    base_initializer: str
    postprocessors: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitializationContext:
    """Provide model-local data and an immutable-base cache to initializers."""

    normalized_mesh: Any
    model_path: Path
    base_cache: "BaseInitializationCache"


@dataclass(frozen=True)
class ProblemBundle:
    """Own the fresh mutable numerical state for one complete benchmark run."""

    normalized_mesh: Any
    mesh_2d: Any
    nodal_variables: Any
    newton_flow_energy: Any
    problem: Any
    optimizer: Any


@dataclass(frozen=True)
class StageOutcome:
    """Summarize one ordinary-Newton or extrapolation stage."""

    completed_iterations: int
    converged: bool
    failed: bool
    termination_reason: str


@dataclass(frozen=True)
class InitialOptimizerResult:
    """Summarize Phase 1 and provide the benchmark bundle for Phase 2."""

    completed_iterations: int
    benchmark_converged: bool
    failed: bool
    termination_reason: str
    bundle: ProblemBundle


@dataclass(frozen=True)
class RunOutcome:
    """Summarize a complete method run, including all of its stages."""

    completed_iterations: int
    converged: bool
    failed: bool
    termination_reason: str


InitializerCreator = Callable[[InitializationContext, InitializerSpec], InitializationResult]
TokenParser = Callable[[str], Optional[MethodSpec]]
MainStageRunner = Callable[[ProblemBundle, RunSpec, "IterationRecorder", int, int], StageOutcome]
InitialOptimizerRunner = Callable[
    [ProblemBundle, RunSpec, "IterationRecorder", int],
    InitialOptimizerResult,
]


@dataclass(frozen=True)
class InitializerAdapter:
    """Define one registered initializer's CLI aliases and construction callback."""

    key: str
    aliases: tuple[str, ...]
    create: InitializerCreator
    add_cli_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None
    build_spec: Optional[Callable[[argparse.Namespace, str], InitializerSpec]] = None


@dataclass(frozen=True)
class MethodAdapter:
    """Define token parsing and numerical execution for one method family."""

    key: str
    token_description: str
    try_parse_token: TokenParser
    uses_initial_optimization: bool
    uses_constant_speed: bool
    run_main_stage: MainStageRunner
    add_cli_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None


@dataclass(frozen=True)
class InitialOptimizerAdapter:
    """Define one registered Phase-1 optimizer and its command-line aliases."""

    key: str
    canonical_name: str
    aliases: tuple[str, ...]
    run_stage: InitialOptimizerRunner
    add_cli_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None


class BaseInitializationCache:
    """Cache immutable base embeddings within one model's initializer matrix."""

    def __init__(self) -> None:
        """Create an empty model-scoped base-initialization cache."""

        self._entries: dict[tuple[Any, ...], BaseInitialization] = {}

    def get_or_create(
        self,
        key: tuple[Any, ...],
        factory: Callable[[], BaseInitialization],
    ) -> BaseInitialization:
        """Return a copied cached base embedding, creating it on first use."""

        if key not in self._entries:
            created = factory()
            uv = np.array(created.uv, dtype=float, copy=True)
            uv.setflags(write=False)
            self._entries[key] = BaseInitialization(
                name=created.name,
                uv=uv,
                metadata=dict(created.metadata),
            )

        cached = self._entries[key]
        return BaseInitialization(
            name=cached.name,
            uv=np.array(cached.uv, copy=True),
            metadata=dict(cached.metadata),
        )


class InitializerRegistry:
    """Resolve initializer tokens without hardcoding names in the experiment loop."""

    def __init__(self) -> None:
        """Create an empty initializer registry."""

        self._adapters: dict[str, InitializerAdapter] = {}
        self._tokens: dict[str, str] = {}

    def register(self, adapter: InitializerAdapter) -> None:
        """Register an initializer adapter and reject duplicate keys or aliases."""

        key = adapter.key.lower()
        if key in self._adapters:
            raise ValueError(f"Duplicate initializer adapter key: {adapter.key}")

        tokens = (adapter.key, *adapter.aliases)
        for token in tokens:
            normalized = token.lower()
            if normalized in self._tokens:
                raise ValueError(f"Duplicate initializer token or alias: {token}")

        self._adapters[key] = adapter
        for token in tokens:
            self._tokens[token.lower()] = key

    def resolve(self, token: str) -> InitializerAdapter:
        """Resolve one case-insensitive initializer token to its adapter."""

        key = self._tokens.get(token.lower())
        if key is None:
            raise ValueError(
                f"Unknown initializer '{token}'. Available presets: {self.help_summary()}"
            )
        return self._adapters[key]

    def adapters(self) -> tuple[InitializerAdapter, ...]:
        """Return registered adapters in deterministic registration order."""

        return tuple(self._adapters.values())

    def help_summary(self) -> str:
        """Return canonical initializer names for command-line help and errors."""

        return ", ".join(adapter.key for adapter in self._adapters.values())


class InitialOptimizerRegistry:
    """Resolve Phase-1 optimizer tokens and dispatch their registered runners."""

    def __init__(self) -> None:
        """Create an empty initial-optimizer registry."""

        self._adapters: dict[str, InitialOptimizerAdapter] = {}
        self._tokens: dict[str, str] = {}

    def register(self, adapter: InitialOptimizerAdapter) -> None:
        """Register one optimizer and reject duplicate keys or aliases."""

        key = adapter.key.lower()
        if key in self._adapters:
            raise ValueError(f"Duplicate initial-optimizer adapter key: {adapter.key}")

        normalized_tokens: list[str] = []
        for token in (adapter.key, adapter.canonical_name, *adapter.aliases):
            normalized = token.lower()
            if normalized in normalized_tokens:
                continue
            if normalized in self._tokens:
                raise ValueError(f"Duplicate initial-optimizer token or alias: {token}")
            normalized_tokens.append(normalized)

        self._adapters[key] = adapter
        for normalized in normalized_tokens:
            self._tokens[normalized] = key

    def resolve(self, token: str) -> InitialOptimizerAdapter:
        """Resolve one case-insensitive optimizer token to its adapter."""

        key = self._tokens.get(token.lower())
        if key is None:
            raise ValueError(
                f"Unknown initial optimizer '{token}'. Available optimizers: "
                f"{self.help_summary()}"
            )
        return self._adapters[key]

    def get(self, key: str) -> InitialOptimizerAdapter:
        """Return the adapter registered under a stable key."""

        try:
            return self._adapters[key.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown initial-optimizer adapter key: {key}") from exc

    def adapters(self) -> tuple[InitialOptimizerAdapter, ...]:
        """Return adapters in deterministic registration order."""

        return tuple(self._adapters.values())

    def help_summary(self) -> str:
        """Return canonical optimizer names for CLI help and errors."""

        return ", ".join(adapter.canonical_name for adapter in self._adapters.values())


class MethodRegistry:
    """Resolve method tokens and dispatch numerical stages through adapters."""

    def __init__(self) -> None:
        """Create an empty method registry."""

        self._adapters: dict[str, MethodAdapter] = {}

    def register(self, adapter: MethodAdapter) -> None:
        """Register a method adapter and reject duplicate keys."""

        key = adapter.key.lower()
        if key in self._adapters:
            raise ValueError(f"Duplicate method adapter key: {adapter.key}")
        self._adapters[key] = adapter

    def parse(self, token: str) -> MethodSpec:
        """Parse one method token and require exactly one adapter match."""

        matches = [
            spec
            for adapter in self._adapters.values()
            if (spec := adapter.try_parse_token(token)) is not None
        ]
        if not matches:
            raise ValueError(
                f"Unknown method '{token}'. Available forms: {self.help_summary()}"
            )
        if len(matches) > 1:
            raise ValueError(f"Ambiguous method token '{token}'")
        return matches[0]

    def get(self, key: str) -> MethodAdapter:
        """Return the adapter registered under a stable key."""

        try:
            return self._adapters[key.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown method adapter key: {key}") from exc

    def adapters(self) -> tuple[MethodAdapter, ...]:
        """Return registered adapters in deterministic registration order."""

        return tuple(self._adapters.values())

    def help_summary(self) -> str:
        """Return registered token descriptions for command-line help and errors."""

        return ", ".join(adapter.token_description for adapter in self._adapters.values())


class CsvIterationWriter:
    """Append iteration rows while enforcing one stable CSV schema."""

    def __init__(self, output_path: Path) -> None:
        """Open an output CSV, create its header, or validate an existing header."""

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        needs_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        if not needs_header:
            with self.output_path.open("r", newline="", encoding="utf-8") as existing:
                header = next(csv.reader(existing), None)
            if tuple(header or ()) != CSV_COLUMNS:
                raise ValueError(
                    f"Existing CSV header does not match the benchmark schema: {self.output_path}"
                )

        self._file = self.output_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._last_positions: dict[str, tuple[int, int]] = {}
        if needs_header:
            self._writer.writeheader()
            self._file.flush()

    def write_row(self, row: Mapping[str, Any]) -> None:
        """Append and flush one complete iteration row."""

        missing = set(CSV_COLUMNS) - set(row)
        extra = set(row) - set(CSV_COLUMNS)
        if missing or extra:
            raise ValueError(f"CSV row schema mismatch; missing={missing}, extra={extra}")
        phase = int(row["Phase"])
        iteration_index = int(row["iteration_index"])
        if phase not in tuple(int(value) for value in BenchmarkPhase):
            raise ValueError(f"Invalid benchmark Phase: {phase}")
        if phase == int(BenchmarkPhase.UV_INITIALIZATION) and iteration_index != 0:
            raise ValueError("Phase 0 requires iteration_index=0")
        if phase != int(BenchmarkPhase.UV_INITIALIZATION) and iteration_index < 1:
            raise ValueError("Optimization phases require iteration_index>=1")
        if int(row["initial_optimization_iters"]) == 0:
            if row["initial_optimizer"] != "N/A":
                raise ValueError("Disabled Phase 1 requires initial_optimizer=N/A")
            if row["initial_optimization_grad_tol"] != "NA":
                raise ValueError("Disabled Phase 1 requires initial_optimization_grad_tol=NA")
        elif row["initial_optimizer"] == "N/A":
            raise ValueError("Enabled Phase 1 requires a named initial optimizer")
        else:
            grad_tol = float(row["initial_optimization_grad_tol"])
            if not np.isfinite(grad_tol) or grad_tol <= 0:
                raise ValueError("Enabled Phase 1 requires a positive finite gradient tolerance")
        if row["method"] == "N/A":
            if (
                row["constant_speed_indicator"] != "NA"
                or phase == int(BenchmarkPhase.CORE_METHOD)
            ):
                raise ValueError("Initial-only rows require constant_speed_indicator=NA and no Phase 2")
        self._writer.writerow(dict(row))
        self._file.flush()
        self._last_positions[str(row["ID"])] = (phase, iteration_index)

    def last_position(self, run_id: UUID | str) -> Optional[tuple[int, int]]:
        """Return the last successfully written phase/index for one run."""

        return self._last_positions.get(str(run_id))

    def close(self) -> None:
        """Flush and close the underlying CSV file."""

        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "CsvIterationWriter":
        """Return this writer as a context manager."""

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close the CSV whether the experiment succeeds or raises."""

        self.close()


class RunStatusCsvWriter:
    """Append one terminal outcome row for each benchmark run specification."""

    def __init__(self, output_path: Path) -> None:
        """Open a status CSV and validate an existing exact header."""

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        if not needs_header:
            with self.output_path.open("r", newline="", encoding="utf-8") as existing:
                header = next(csv.reader(existing), None)
            if tuple(header or ()) != RUN_STATUS_COLUMNS:
                raise ValueError(
                    "Existing run-status CSV header does not match the benchmark "
                    f"schema: {self.output_path}"
                )

        self._file = self.output_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=RUN_STATUS_COLUMNS)
        self._written_ids: set[str] = set()
        if needs_header:
            self._writer.writeheader()
            self._file.flush()

    def write_row(self, row: Mapping[str, Any]) -> None:
        """Validate, append, and flush one terminal run-status row."""

        missing = set(RUN_STATUS_COLUMNS) - set(row)
        extra = set(row) - set(RUN_STATUS_COLUMNS)
        if missing or extra:
            raise ValueError(
                f"Run-status row schema mismatch; missing={missing}, extra={extra}"
            )

        run_id = str(row["ID"])
        UUID(run_id)
        if run_id in self._written_ids:
            raise ValueError(f"Duplicate run-status row for ID={run_id}")

        status = str(row["status"])
        failure_scope = str(row["failure_scope"])
        if status not in RUN_STATUS_VALUES:
            raise ValueError(f"Invalid run status: {status}")
        if failure_scope not in FAILURE_SCOPE_VALUES:
            raise ValueError(f"Invalid failure scope: {failure_scope}")
        if status in {"converged", "completed"} and failure_scope != "NA":
            raise ValueError("Successful run statuses require failure_scope=NA")
        if status == "skipped" and failure_scope not in {"model_load", "initializer"}:
            raise ValueError("Skipped status requires a model or initializer failure scope")
        if status == "failed" and failure_scope != "run_execution":
            raise ValueError("Failed status requires failure_scope=run_execution")

        initial_iters = int(row["initial_optimization_iters"])
        initial_optimizer = row["initial_optimizer"]
        initial_tol = row["initial_optimization_grad_tol"]
        if initial_iters == 0:
            if initial_optimizer != "N/A" or initial_tol != "NA":
                raise ValueError(
                    "Disabled Phase 1 requires N/A optimizer and gradient tolerance"
                )
        else:
            if initial_optimizer == "N/A":
                raise ValueError("Enabled Phase 1 requires a named initial optimizer")
            grad_tol = float(initial_tol)
            if not np.isfinite(grad_tol) or grad_tol <= 0:
                raise ValueError(
                    "Enabled Phase 1 requires a positive finite gradient tolerance"
                )
        if row["method"] == "N/A" and row["constant_speed_indicator"] != "NA":
            raise ValueError("Initial-only status rows require constant speed NA")

        completed = int(row["completed_iterations"])
        if completed < 0:
            raise ValueError("completed_iterations must be nonnegative")
        if completed > int(row["max_iters"]):
            raise ValueError("completed_iterations cannot exceed max_iters")
        phase = row["last_recorded_phase"]
        iteration = row["last_recorded_iteration"]
        if (phase == "NA") != (iteration == "NA"):
            raise ValueError("Last recorded phase and iteration must both be NA or numeric")
        if phase != "NA":
            phase = int(phase)
            iteration = int(iteration)
            if phase not in tuple(int(value) for value in BenchmarkPhase):
                raise ValueError(f"Invalid last recorded phase: {phase}")
            if iteration < 0:
                raise ValueError("last_recorded_iteration must be nonnegative")
            if phase == int(BenchmarkPhase.UV_INITIALIZATION) and iteration != 0:
                raise ValueError("A Phase 0 last position requires iteration zero")
            if phase != int(BenchmarkPhase.UV_INITIALIZATION) and iteration < 1:
                raise ValueError("An optimization last position requires iteration >= 1")
        if status == "skipped" and (
            completed != 0 or phase != "NA" or iteration != "NA"
        ):
            raise ValueError("Skipped runs cannot have completed or recorded iterations")
        if phase != "NA" and completed != iteration:
            raise ValueError(
                "completed_iterations must equal the last continuous iteration index"
            )

        exception_type = row["exception_type"]
        exception_message = row["exception_message"]
        if (exception_type == "NA") != (exception_message == "NA"):
            raise ValueError("Exception type and message must both be NA or both be present")
        if status in {"converged", "completed"} and exception_type != "NA":
            raise ValueError("Successful run statuses cannot contain an exception")
        if not str(row["termination_reason"]):
            raise ValueError("termination_reason must not be empty")

        finished = datetime.fromisoformat(
            str(row["finished_timestamp"]).replace("Z", "+00:00")
        )
        if finished.tzinfo is None:
            raise ValueError("finished_timestamp must include a UTC offset")

        self._writer.writerow(dict(row))
        self._file.flush()
        self._written_ids.add(run_id)

    def close(self) -> None:
        """Flush and close the underlying status CSV."""

        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "RunStatusCsvWriter":
        """Return this writer as a context manager."""

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close the status CSV whether the benchmark succeeds or raises."""

        self.close()


class IterationRecorder:
    """Collect metrics and write rows for one complete benchmark run."""

    def __init__(
        self,
        writer: CsvIterationWriter,
        run_spec: RunSpec,
    ) -> None:
        """Bind a CSV writer and immutable run identity to iteration recording."""

        self.writer = writer
        self.run_spec = run_spec
        self.nonfinite_detected = False

    def record(
        self,
        problem: Any,
        phase: BenchmarkPhase,
        iteration_index: int,
        elapsed_ns: int,
    ) -> bool:
        """Record one benchmark state and return whether its metrics are finite."""

        energy_value, gradient_norm = collect_iteration_metrics(problem)
        finite = bool(np.isfinite(energy_value) and np.isfinite(gradient_norm))
        self.nonfinite_detected = self.nonfinite_detected or not finite
        self.writer.write_row(
            build_iteration_row(
                run_spec=self.run_spec,
                phase=phase,
                iteration_index=iteration_index,
                energy_value=energy_value,
                gradient_norm=gradient_norm,
                elapsed_seconds=elapsed_ns / 1_000_000_000.0,
            )
        )
        return finite


class _OrdinaryNewtonCallback:
    """Translate Newton's pre-iteration callback into completed-step records."""

    def __init__(
        self,
        recorder: IterationRecorder,
        phase: BenchmarkPhase,
        iteration_offset: int,
    ) -> None:
        """Create a timer whose first callback starts and later callbacks close steps."""

        self.recorder = recorder
        self.phase = phase
        self.iteration_offset = iteration_offset
        self.completed_iterations = 0
        self._start_ns: Optional[int] = None

    def __call__(self, problem: Any, callback_index: int) -> bool:
        """Record the preceding Newton step, then start timing the next one."""

        del callback_index
        now_ns = time.perf_counter_ns()
        should_stop = False
        if self._start_ns is not None:
            iteration_index = self.iteration_offset + self.completed_iterations + 1
            finite = self.recorder.record(
                problem,
                self.phase,
                iteration_index,
                now_ns - self._start_ns,
            )
            self.completed_iterations += 1
            should_stop = not finite

        self._start_ns = time.perf_counter_ns()
        return should_stop


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with a trailing Z."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def run_status_csv_path(output_csv: Path) -> Path:
    """Derive the automatic companion status path for an iteration CSV."""

    output_csv = Path(output_csv)
    return output_csv.with_name(f"{output_csv.stem}_run_status.csv")


def build_run_status_row(
    run_spec: RunSpec,
    status: str,
    termination_reason: str,
    failure_scope: str,
    completed_iterations: int,
    last_position: Optional[tuple[int, int]],
    exception: Optional[BaseException] = None,
) -> dict[str, Any]:
    """Build one terminal row matching the companion run-status contract."""

    constant_speed = (
        None if run_spec.method_spec is None else run_spec.method_spec.constant_speed
    )
    if last_position is None:
        last_phase: int | str = "NA"
        last_iteration: int | str = "NA"
    else:
        last_phase, last_iteration = last_position
    return {
        "ID": str(run_spec.run_id),
        "model_name": run_spec.model_name,
        "method": run_spec.method_name,
        "initializer": run_spec.initializer_spec.canonical_name,
        "initial_optimizer": run_spec.initial_optimizer_name,
        "initial_optimization_iters": run_spec.initial_optimization_iters,
        "initial_optimization_grad_tol": (
            "NA" if run_spec.initial_optimization_grad_tol is None
            else run_spec.initial_optimization_grad_tol
        ),
        "repeat_index": run_spec.repeat_index,
        "constant_speed_indicator": (
            "NA" if constant_speed is None else str(bool(constant_speed))
        ),
        "thread_num": run_spec.thread_num,
        "max_iters": run_spec.max_iters,
        "status": status,
        "termination_reason": termination_reason,
        "failure_scope": failure_scope,
        "completed_iterations": completed_iterations,
        "last_recorded_phase": last_phase,
        "last_recorded_iteration": last_iteration,
        "exception_type": "NA" if exception is None else type(exception).__name__,
        "exception_message": "NA" if exception is None else str(exception),
        "finished_timestamp": utc_timestamp(),
    }


def collect_iteration_metrics(problem: Any) -> tuple[float, float]:
    """Evaluate energy and full Euclidean gradient norm at the current iterate."""

    energy_value = float(problem.energy())
    gradient_norm = float(np.linalg.norm(np.asarray(problem.gradient())))
    return energy_value, gradient_norm


def build_iteration_row(
    run_spec: RunSpec,
    phase: BenchmarkPhase,
    iteration_index: int,
    energy_value: float,
    gradient_norm: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build one row matching the public phase-aware CSV contract."""

    constant_speed = (
        None if run_spec.method_spec is None else run_spec.method_spec.constant_speed
    )
    speed_indicator = "NA" if constant_speed is None else str(bool(constant_speed))
    return {
        "ID": str(run_spec.run_id),
        "model_name": run_spec.model_name,
        "method": run_spec.method_name,
        "initializer": run_spec.initializer_spec.canonical_name,
        "initial_optimizer": run_spec.initial_optimizer_name,
        "initial_optimization_iters": run_spec.initial_optimization_iters,
        "initial_optimization_grad_tol": (
            "NA" if run_spec.initial_optimization_grad_tol is None
            else run_spec.initial_optimization_grad_tol
        ),
        "repeat_index": run_spec.repeat_index,
        "constant_speed_indicator": speed_indicator,
        "thread_num": run_spec.thread_num,
        "Phase": int(phase),
        "iteration_index": iteration_index,
        "energy": energy_value,
        "gradient_norm": gradient_norm,
        "time_spent_within_this_iteration": elapsed_seconds,
        "current_timestamp": utc_timestamp(),
    }


def discover_models(models_dir: Path, recursive: bool = False) -> list[Path]:
    """Discover supported mesh files, optionally including subdirectories."""

    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        raise ValueError(f"Models directory does not exist or is not a directory: {models_dir}")

    candidates = models_dir.rglob("*") if recursive else models_dir.iterdir()
    model_paths = sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_EXTENSIONS
        ),
        key=lambda path: (
            str(path.relative_to(models_dir)).lower(),
            str(path.relative_to(models_dir)),
        ),
    )
    if not model_paths:
        raise ValueError(f"No OBJ, OFF, or MSH models found in: {models_dir}")

    if not recursive:
        stems: dict[str, Path] = {}
        for path in model_paths:
            normalized_stem = path.stem.lower()
            if normalized_stem in stems:
                raise ValueError(
                    "Duplicate model stem in non-recursive discovery: "
                    f"{stems[normalized_stem].name} and {path.name}"
                )
            stems[normalized_stem] = path
    return model_paths


def load_normalized_mesh(model_path: Path) -> Any:
    """Load a triangle mesh and normalize its total surface area to one."""

    model_path = Path(model_path)
    original_mesh = param_utils.load(str(model_path))
    vertices = np.asarray(original_mesh.vertices(), dtype=float)
    elements = np.asarray(original_mesh.elements(), dtype=int)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        raise ValueError(f"Model has no valid vertices: {model_path}")
    if elements.ndim != 2 or elements.shape[0] == 0 or elements.shape[1] != 3:
        raise ValueError(f"Model is not a nonempty triangle mesh: {model_path}")

    total_area = float(np.sum(np.asarray(original_mesh.elementVolumes(), dtype=float)))
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError(f"Model has non-positive or non-finite surface area: {model_path}")
    return mesh.Mesh(vertices / np.sqrt(total_area), elements)


def validate_uv_array(uv: np.ndarray, num_vertices: int, initializer_name: str) -> np.ndarray:
    """Validate and copy an initializer's numVertices-by-2 UV array."""

    uv_array = np.asarray(uv)
    if np.iscomplexobj(uv_array):
        raise ValueError(f"Initializer {initializer_name} produced complex UV values")
    uv_array = np.array(uv_array, dtype=float, copy=True)
    if uv_array.shape != (num_vertices, 2):
        raise ValueError(
            f"Initializer {initializer_name} produced shape {uv_array.shape}; "
            f"expected {(num_vertices, 2)}"
        )
    if not np.all(np.isfinite(uv_array)):
        raise ValueError(f"Initializer {initializer_name} produced non-finite UV values")
    return uv_array


def generate_tutte_base(context: InitializationContext) -> BaseInitialization:
    """Generate the notebook's normalized-circle Tutte embedding with flip fallback."""

    normalized_mesh = context.normalized_mesh
    boundary_uv = helper_funcs.getBDdataOnNormalizedCircle(normalized_mesh)
    initial_uv = parametrization.harmonic(normalized_mesh, boundary_uv, False)
    initial_flip_count = len(parametrization.getFlips(normalized_mesh, initial_uv))
    used_uniform_weights = initial_flip_count > 0
    if used_uniform_weights:
        initial_uv = parametrization.harmonic(normalized_mesh, boundary_uv, True)

    final_uv = validate_uv_array(initial_uv, normalized_mesh.numVertices(), "tutte")
    final_flip_count = len(parametrization.getFlips(normalized_mesh, final_uv))
    if final_flip_count > 0:
        raise ValueError(
            f"Tutte fallback still contains {final_flip_count} flipped triangles for "
            f"{context.model_path.name}"
        )
    return BaseInitialization(
        name="tutte",
        uv=final_uv,
        metadata={
            "initial_flip_count": initial_flip_count,
            "final_flip_count": final_flip_count,
            "used_uniform_weights": used_uniform_weights,
        },
    )


def get_tutte_base(context: InitializationContext) -> BaseInitialization:
    """Return an independent copy of the model's cached Tutte base embedding."""

    return context.base_cache.get_or_create(
        ("tutte",),
        lambda: generate_tutte_base(context),
    )


def compute_initialization_scale(
    context: InitializationContext,
    base_uv: np.ndarray,
    strategy_name: str,
) -> float:
    """Compute and validate one initial_utils global scale on temporary state."""

    normalized_mesh = context.normalized_mesh
    temporary_uv = mesh_energy.NodalVars(normalized_mesh, 2)
    temporary_uv.setVars(np.asarray(base_uv).ravel())
    temporary_param = continuation_parametrization.symmetric_dirichlet_param(
        normalized_mesh,
        temporary_uv,
    )
    raw_scale = initial_utils.initialization_scale(
        normalized_mesh,
        temporary_uv,
        temporary_param,
        strategy_name,
    )
    if np.iscomplexobj(raw_scale):
        raise ValueError(
            f"Initializer {strategy_name} produced a complex scale for {context.model_path.name}"
        )
    scale = float(raw_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"Initializer {strategy_name} produced invalid scale {scale} for "
            f"{context.model_path.name}"
        )
    return scale


def create_tutte_initialization(
    context: InitializationContext,
    spec: InitializerSpec,
) -> InitializationResult:
    """Create the identity Tutte initializer result."""

    base = get_tutte_base(context)
    return InitializationResult(
        name=spec.canonical_name,
        uv=validate_uv_array(base.uv, context.normalized_mesh.numVertices(), spec.canonical_name),
        base_initializer="tutte",
        postprocessors=(),
        metadata={**base.metadata, "scale_strategy": "orig_tutte", "scale_factor": 1.0},
    )


def create_scaled_tutte_initialization(
    context: InitializationContext,
    spec: InitializerSpec,
) -> InitializationResult:
    """Create a registered Tutte-plus-global-scale initializer result."""

    strategy_name = spec.canonical_name
    base = get_tutte_base(context)
    scale = compute_initialization_scale(context, base.uv, strategy_name)
    finalized_uv = validate_uv_array(
        scale * base.uv,
        context.normalized_mesh.numVertices(),
        strategy_name,
    )
    return InitializationResult(
        name=strategy_name,
        uv=finalized_uv,
        base_initializer="tutte",
        postprocessors=("global_scale",),
        metadata={
            **base.metadata,
            "scale_strategy": strategy_name,
            "scale_factor": scale,
        },
    )


def validate_initialization_result(
    result: InitializationResult,
    context: InitializationContext,
    spec: InitializerSpec,
) -> InitializationResult:
    """Validate canonical naming and finalized UV ownership for an initializer result."""

    if result.name != spec.canonical_name:
        raise ValueError(
            f"Initializer adapter '{spec.adapter_key}' returned name '{result.name}', "
            f"expected '{spec.canonical_name}'"
        )
    uv = validate_uv_array(
        result.uv,
        context.normalized_mesh.numVertices(),
        spec.canonical_name,
    )
    return InitializationResult(
        name=result.name,
        uv=uv,
        base_initializer=result.base_initializer,
        postprocessors=tuple(result.postprocessors),
        metadata=dict(result.metadata),
    )


def initialize_uv(
    context: InitializationContext,
    spec: InitializerSpec,
    registry: InitializerRegistry,
) -> InitializationResult:
    """Dispatch and validate one initializer through the registered adapter."""

    adapter = registry.resolve(spec.adapter_key)
    result = adapter.create(context, spec)
    return validate_initialization_result(result, context, spec)


def timed_initialize_uv(
    context: InitializationContext,
    spec: InitializerSpec,
    registry: InitializerRegistry,
) -> tuple[InitializationResult, int]:
    """Create one initialization and return its elapsed wall time in nanoseconds."""

    start_ns = time.perf_counter_ns()
    result = initialize_uv(context, spec, registry)
    return result, time.perf_counter_ns() - start_ns


def register_builtin_initializers() -> InitializerRegistry:
    """Register Tutte and every current initial_utils scale-derived preset."""

    registry = InitializerRegistry()
    registry.register(
        InitializerAdapter(
            key="tutte",
            aliases=("orig_tutte",),
            create=create_tutte_initialization,
        )
    )
    for name in SCALE_INITIALIZER_NAMES:
        registry.register(
            InitializerAdapter(
                key=name,
                aliases=(),
                create=create_scaled_tutte_initialization,
            )
        )
    return registry


def parse_initializer_specs(
    args: argparse.Namespace,
    registry: InitializerRegistry,
) -> list[InitializerSpec]:
    """Resolve the ordered --initializers list and reject canonical duplicates."""

    tokens = tuple(args.initializers or ("tutte",))
    specs: list[InitializerSpec] = []
    seen_names: set[str] = set()
    for token in tokens:
        adapter = registry.resolve(token)
        if adapter.build_spec is None:
            spec = InitializerSpec(adapter_key=adapter.key, canonical_name=adapter.key)
        else:
            spec = adapter.build_spec(args, token)
        if spec.canonical_name in seen_names:
            raise ValueError(
                f"Duplicate initializer after normalization: {spec.canonical_name}"
            )
        seen_names.add(spec.canonical_name)
        specs.append(spec)
    return specs


def parse_initial_optimizer_specs(
    args: argparse.Namespace,
    registry: InitialOptimizerRegistry,
) -> list[InitialOptimizerSpec]:
    """Resolve --initial-optimizer tokens and reject canonical duplicates."""

    tokens = tuple(args.initial_optimizer or ("Newton",))
    specs: list[InitialOptimizerSpec] = []
    seen_names: set[str] = set()
    for token in tokens:
        adapter = registry.resolve(token)
        if adapter.canonical_name in seen_names:
            raise ValueError(
                "Duplicate initial optimizer after normalization: "
                f"{adapter.canonical_name}"
            )
        seen_names.add(adapter.canonical_name)
        specs.append(
            InitialOptimizerSpec(
                adapter_key=adapter.key,
                canonical_name=adapter.canonical_name,
            )
        )
    return specs


def parse_newton_token(token: str) -> Optional[MethodSpec]:
    """Parse the exact Newton method token."""

    if token.lower() != "newton":
        return None
    return MethodSpec(adapter_key="newton", display_name="Newton")


def parse_slim_token(token: str) -> Optional[MethodSpec]:
    """Parse the case-insensitive SLIM method token."""

    if token.lower() != "slim":
        return None
    return MethodSpec(adapter_key="slim", display_name="SLIM")


def parse_rs_token(token: str) -> Optional[MethodSpec]:
    """Parse the exact rotation-strain method token."""

    if token.lower() != "rs":
        return None
    return MethodSpec(adapter_key="rs", display_name="RS")


def _parse_degree_method_token(
    token: str,
    family_name: str,
    adapter_key: str,
) -> Optional[MethodSpec]:
    """Parse and validate one degree-parameterized Taylor or Pade token."""

    match = re.fullmatch(rf"{family_name}(\d+)", token, flags=re.IGNORECASE)
    if match is None:
        return None
    degree = int(match.group(1))
    if not 1 <= degree <= 20:
        raise ValueError(f"{family_name} degree must be between 1 and 20: {token}")
    return MethodSpec(
        adapter_key=adapter_key,
        display_name=f"{family_name}{degree}",
        options={"degree": degree},
    )


def parse_taylor_token(token: str) -> Optional[MethodSpec]:
    """Parse a TaylorN token with a supported degree from 1 through 20."""

    return _parse_degree_method_token(token, "Taylor", "taylor")


def parse_pade_token(token: str) -> Optional[MethodSpec]:
    """Parse a PadeN token with a supported degree from 1 through 20."""

    return _parse_degree_method_token(token, "Pade", "pade")


def expand_method_specs(
    tokens: Sequence[str],
    constant_speed_mode: Optional[str],
    registry: MethodRegistry,
) -> list[MethodSpec]:
    """Parse methods, reject duplicates, and expand constant-speed variants."""

    parsed_specs: list[MethodSpec] = []
    seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
    for token in tokens:
        spec = registry.parse(token)
        identity = (spec.adapter_key, tuple(sorted(spec.options.items())))
        if identity in seen:
            raise ValueError(f"Duplicate method after normalization: {spec.display_name}")
        seen.add(identity)
        parsed_specs.append(spec)

    requires_speed = any(registry.get(spec.adapter_key).uses_constant_speed for spec in parsed_specs)
    if requires_speed and constant_speed_mode is None:
        raise ValueError(
            "--constant-speed is required when a selected method uses Newton-flow speed"
        )

    expanded: list[MethodSpec] = []
    for spec in parsed_specs:
        adapter = registry.get(spec.adapter_key)
        if not adapter.uses_constant_speed:
            expanded.append(spec)
            continue
        speed_values = {
            "true": (True,),
            "false": (False,),
            "both": (True, False),
        }[str(constant_speed_mode).lower()]
        expanded.extend(
            MethodSpec(
                adapter_key=spec.adapter_key,
                display_name=spec.display_name,
                options=dict(spec.options),
                constant_speed=speed,
            )
            for speed in speed_values
        )
    return expanded


def make_golden_section_search() -> opt_utils.GoldenSectionSearch:
    """Create the notebook's golden-section line search configuration."""

    return opt_utils.GoldenSectionSearch(max_alpha=20, golden_tol=0.1)


def build_common_problem_bundle(
    normalized_mesh: Any,
    initial_uv: np.ndarray,
    max_iters: int,
) -> ProblemBundle:
    """Construct fresh Newton-flow problem and optimizer state for one run."""

    nodal_variables = mesh_energy.NodalVars(normalized_mesh, 2)
    nodal_variables.setVars(np.asarray(initial_uv).ravel())

    mesh_2d = mesh.Mesh(
        np.zeros((normalized_mesh.numVertices(), 2)),
        normalized_mesh.elements(),
    )
    mesh_2d.reembedElements(normalized_mesh.vertices())
    flow_energy = fast_newton_flow.symmetric_dirichlet(mesh_2d, nodal_variables)
    problem = py_newton_optimizer.NewtonMultiobjectiveProblem(
        nodal_variables,
        [flow_energy],
    )

    flow_energy.elementHessianShift = 1e-10
    problem.hessianShift = 0
    problem.useRelativeHessianShift = False
    problem.initialFeasibleStepLengthComputer = (
        flip_avoiding_step_length.FlipAvoidingStepLength(normalized_mesh.elements())
    )
    problem.initialFeasibleStepLengthComputer.backoffFactor = 0.8

    optimizer = problem.optimizer()
    optimizer.options.hessianProjectionController = (
        py_newton_optimizer.HessianProjectionAdaptive()
    )
    optimizer.options.hessianProjectionController.startWithProjectionActive = False
    optimizer.options.hessianProjectionController.numProjectionStepsBeforeDisable = 2
    optimizer.options.hessianProjectionController.numConsecutiveIndefiniteStepsBeforeEnable = 0
    optimizer.options.niter = max_iters
    optimizer.options.gradTol = GRADIENT_TOLERANCE
    optimizer.options.verbose = 0

    return ProblemBundle(
        normalized_mesh=normalized_mesh,
        mesh_2d=mesh_2d,
        nodal_variables=nodal_variables,
        newton_flow_energy=flow_energy,
        problem=problem,
        optimizer=optimizer,
    )


def _final_gradient_norm(problem: Any) -> float:
    """Evaluate the final full gradient norm for stage termination classification."""

    return float(np.linalg.norm(np.asarray(problem.gradient())))


def run_ordinary_newton_stage(
    bundle: ProblemBundle,
    recorder: IterationRecorder,
    phase: BenchmarkPhase,
    iteration_offset: int,
    max_iters: int,
    grad_tol: float = GRADIENT_TOLERANCE,
) -> StageOutcome:
    """Run and record an ordinary-Newton stage through MeshFEM's optimizer."""

    if max_iters <= 0:
        return StageOutcome(0, False, False, "iteration_budget_exhausted")

    callback = _OrdinaryNewtonCallback(recorder, phase, iteration_offset)
    bundle.problem.setCustomIterationCallback(callback)
    bundle.optimizer.options.niter = max_iters
    bundle.optimizer.options.gradTol = grad_tol
    bundle.optimizer.optimize()
    completed = callback.completed_iterations
    final_gradient = _final_gradient_norm(bundle.problem)

    if recorder.nonfinite_detected or not np.isfinite(final_gradient):
        return StageOutcome(completed, False, True, "nonfinite_state")
    if completed >= max_iters:
        return StageOutcome(completed, False, False, "iteration_budget_exhausted")
    if final_gradient < grad_tol:
        return StageOutcome(completed, True, False, "gradient_tolerance")
    return StageOutcome(completed, False, True, "optimizer_stopped_before_budget")


def run_slim_stage(
    bundle: ProblemBundle,
    recorder: IterationRecorder,
    phase: BenchmarkPhase,
    iteration_offset: int,
    max_iters: int,
    grad_tol: float = GRADIENT_TOLERANCE,
) -> StageOutcome:
    """Run source-reference SLIM with the plan-defined PP-style line search."""

    if max_iters <= 0:
        return StageOutcome(0, False, False, "iteration_budget_exhausted")

    slim_energy = continuation_parametrization.slim_param(
        bundle.normalized_mesh,
        bundle.nodal_variables,
    )
    slim_energy.elementHessianShift = 0.0
    slim_problem = py_newton_optimizer.NewtonMultiobjectiveProblem(
        bundle.nodal_variables,
        [slim_energy],
    )
    slim_problem.setFixedVars([0, 1])
    slim_problem.hessianShift = 0.0
    slim_problem.useRelativeHessianShift = False

    slim_optimizer = slim_problem.optimizer()
    slim_optimizer.options.hessianProjectionController = (
        py_newton_optimizer.HessianProjectionNever()
    )
    slim_optimizer.options.gradTol = grad_tol
    slim_optimizer.options.verbose = 0

    step_limiter = flip_avoiding_step_length.FlipAvoidingStepLength(
        bundle.normalized_mesh.elements()
    )
    step_limiter.backoffFactor = 1.0
    slim_problem.initialFeasibleStepLengthComputer = step_limiter
    line_search = opt_utils.BacktrackArmijoLineSearch(
        backoff_factor=0.8,
        backtrack_factor=0.5,
        armijo_c=0.2,
        max_alpha=1.25,
        step_limiter=step_limiter,
    )
    linear_extrapolator = extra_utils.LinearExtrapolator()
    completed = 0

    while completed < max_iters:
        gradient_norm = _final_gradient_norm(bundle.problem)
        if not np.isfinite(gradient_norm):
            return StageOutcome(completed, False, True, "nonfinite_state")
        if gradient_norm < grad_tol:
            return StageOutcome(completed, True, False, "gradient_tolerance")

        start_ns = time.perf_counter_ns()
        opt_utils.newton_extrapolate(
            slim_optimizer,
            linear_extrapolator,
            line_search,
            grad_tol=0.0,
            max_iters=1,
            verbose=False,
            max_extraNewton_stop_counter=np.inf,
        )
        bundle.problem.setVars(slim_problem.getVars())
        end_ns = time.perf_counter_ns()

        finite = recorder.record(
            bundle.problem,
            phase,
            iteration_offset + completed + 1,
            end_ns - start_ns,
        )
        completed += 1
        if not finite:
            return StageOutcome(completed, False, True, "nonfinite_state")

    final_gradient = _final_gradient_norm(bundle.problem)
    if not np.isfinite(final_gradient):
        return StageOutcome(completed, False, True, "nonfinite_state")
    return StageOutcome(completed, False, False, "iteration_budget_exhausted")


def run_extrapolated_stage(
    bundle: ProblemBundle,
    recorder: IterationRecorder,
    extrapolator: Any,
    iteration_offset: int,
    max_iters: int,
) -> StageOutcome:
    """Run and record a Taylor, Pade, or RS golden-section stage."""

    if max_iters <= 0:
        return StageOutcome(0, False, False, "iteration_budget_exhausted")

    completed = 0
    start_ns = time.perf_counter_ns()

    def post_step_callback(problem: Any, local_index: int, alpha: float) -> None:
        """Close one extrapolation timer and append its completed-iteration row."""

        nonlocal completed, start_ns
        del alpha
        end_ns = time.perf_counter_ns()
        finite = recorder.record(
            problem,
            BenchmarkPhase.CORE_METHOD,
            iteration_offset + local_index + 1,
            end_ns - start_ns,
        )
        completed += 1
        start_ns = time.perf_counter_ns()
        if not finite:
            raise FloatingPointError(
                "Non-finite state in "
                f"{recorder.run_spec.method_spec.display_name} iteration "
                f"{iteration_offset + local_index + 1}"
            )

    opt_utils.newton_extrapolate(
        bundle.optimizer,
        extrapolator,
        make_golden_section_search(),
        post_step_cb=post_step_callback,
        grad_tol=GRADIENT_TOLERANCE,
        max_iters=max_iters,
        verbose=False,
        max_extraNewton_stop_counter=np.inf,
    )
    final_gradient = _final_gradient_norm(bundle.problem)
    if recorder.nonfinite_detected or not np.isfinite(final_gradient):
        return StageOutcome(completed, False, True, "nonfinite_state")
    if final_gradient < GRADIENT_TOLERANCE:
        return StageOutcome(completed, True, False, "gradient_tolerance")
    if completed >= max_iters:
        return StageOutcome(completed, False, False, "iteration_budget_exhausted")
    return StageOutcome(completed, False, True, "extrapolator_stopped_before_budget")


def run_newton_main_stage(
    bundle: ProblemBundle,
    run_spec: RunSpec,
    recorder: IterationRecorder,
    iteration_offset: int,
    max_iters: int,
) -> StageOutcome:
    """Run standalone ordinary Newton for its complete iteration budget."""

    return run_ordinary_newton_stage(
        bundle,
        recorder,
        BenchmarkPhase.CORE_METHOD,
        iteration_offset,
        max_iters,
    )


def run_slim_main_stage(
    bundle: ProblemBundle,
    run_spec: RunSpec,
    recorder: IterationRecorder,
    iteration_offset: int,
    max_iters: int,
) -> StageOutcome:
    """Run source-reference SLIM for the core-method iteration budget."""

    return run_slim_stage(
        bundle,
        recorder,
        BenchmarkPhase.CORE_METHOD,
        iteration_offset,
        max_iters,
    )


def run_taylor_main_stage(
    bundle: ProblemBundle,
    run_spec: RunSpec,
    recorder: IterationRecorder,
    iteration_offset: int,
    max_iters: int,
) -> StageOutcome:
    """Construct and run the requested Taylor Newton-flow extrapolator."""

    extrapolator = extra_utils.TaylorExtrapolator(
        bundle.optimizer,
        int(run_spec.method_spec.options["degree"]),
        constant_speed=bool(run_spec.method_spec.constant_speed),
    )
    return run_extrapolated_stage(
        bundle,
        recorder,
        extrapolator,
        iteration_offset,
        max_iters,
    )


def run_pade_main_stage(
    bundle: ProblemBundle,
    run_spec: RunSpec,
    recorder: IterationRecorder,
    iteration_offset: int,
    max_iters: int,
) -> StageOutcome:
    """Construct and run the requested Pade Newton-flow extrapolator."""

    extrapolator = extra_utils.PadeExtrapolator(
        bundle.optimizer,
        int(run_spec.method_spec.options["degree"]),
        constant_speed=bool(run_spec.method_spec.constant_speed),
    )
    return run_extrapolated_stage(
        bundle,
        recorder,
        extrapolator,
        iteration_offset,
        max_iters,
    )


def run_rs_main_stage(
    bundle: ProblemBundle,
    run_spec: RunSpec,
    recorder: IterationRecorder,
    iteration_offset: int,
    max_iters: int,
) -> StageOutcome:
    """Construct and run the notebook's native rotation-strain extrapolator."""

    extrapolator = rotation_strain_extrapolation.RSNewtonFlowExtrapolator(bundle.mesh_2d)
    return run_extrapolated_stage(
        bundle,
        recorder,
        extrapolator,
        iteration_offset,
        max_iters,
    )


def register_builtin_methods() -> MethodRegistry:
    """Register Newton, SLIM, Taylor, Pade, and rotation-strain adapters."""

    registry = MethodRegistry()
    registry.register(
        MethodAdapter(
            key="newton",
            token_description="Newton",
            try_parse_token=parse_newton_token,
            uses_initial_optimization=True,
            uses_constant_speed=False,
            run_main_stage=run_newton_main_stage,
        )
    )
    registry.register(
        MethodAdapter(
            key="slim",
            token_description="SLIM",
            try_parse_token=parse_slim_token,
            uses_initial_optimization=True,
            uses_constant_speed=False,
            run_main_stage=run_slim_main_stage,
        )
    )
    registry.register(
        MethodAdapter(
            key="taylor",
            token_description="TaylorN (1 <= N <= 20)",
            try_parse_token=parse_taylor_token,
            uses_initial_optimization=True,
            uses_constant_speed=True,
            run_main_stage=run_taylor_main_stage,
        )
    )
    registry.register(
        MethodAdapter(
            key="pade",
            token_description="PadeN (1 <= N <= 20)",
            try_parse_token=parse_pade_token,
            uses_initial_optimization=True,
            uses_constant_speed=True,
            run_main_stage=run_pade_main_stage,
        )
    )
    registry.register(
        MethodAdapter(
            key="rs",
            token_description="RS",
            try_parse_token=parse_rs_token,
            uses_initial_optimization=True,
            uses_constant_speed=False,
            run_main_stage=run_rs_main_stage,
        )
    )
    return registry


def run_newton_initial_optimizer(
    bundle: ProblemBundle,
    run_spec: RunSpec,
    recorder: IterationRecorder,
    max_iters: int,
) -> InitialOptimizerResult:
    """Run Newton in Phase 1 under the common benchmark stopping policy."""

    outcome = run_ordinary_newton_stage(
        bundle,
        recorder,
        BenchmarkPhase.INITIAL_OPTIMIZATION,
        iteration_offset=0,
        max_iters=max_iters,
        grad_tol=run_spec.initial_optimization_grad_tol,
    )
    return InitialOptimizerResult(
        completed_iterations=outcome.completed_iterations,
        benchmark_converged=outcome.converged,
        failed=outcome.failed,
        termination_reason=outcome.termination_reason,
        bundle=bundle,
    )


def run_slim_initial_optimizer(
    bundle: ProblemBundle,
    run_spec: RunSpec,
    recorder: IterationRecorder,
    max_iters: int,
) -> InitialOptimizerResult:
    """Run source-reference SLIM in Phase 1 and hand off its accepted UV state."""

    outcome = run_slim_stage(
        bundle,
        recorder,
        BenchmarkPhase.INITIAL_OPTIMIZATION,
        iteration_offset=0,
        max_iters=max_iters,
        grad_tol=run_spec.initial_optimization_grad_tol,
    )
    bundle.optimizer.options.hessianProjectionController.reset()
    return InitialOptimizerResult(
        completed_iterations=outcome.completed_iterations,
        benchmark_converged=outcome.converged,
        failed=outcome.failed,
        termination_reason=outcome.termination_reason,
        bundle=bundle,
    )


def run_pp_true_area_initial_optimizer(
    bundle: ProblemBundle,
    run_spec: RunSpec,
    recorder: IterationRecorder,
    max_iters: int,
) -> InitialOptimizerResult:
    """Run PP on the selected Phase-0 UVs and stream accepted states into Phase 1."""

    if max_iters <= 0:
        return InitialOptimizerResult(0, False, False, "iteration_budget_exhausted", bundle)
    initial_gradient = _final_gradient_norm(bundle.problem)
    if not np.isfinite(initial_gradient):
        return InitialOptimizerResult(0, False, True, "nonfinite_state", bundle)
    if initial_gradient < run_spec.initial_optimization_grad_tol:
        return InitialOptimizerResult(0, True, False, "gradient_tolerance", bundle)

    from pp_study import PP_utils

    PP_utils._validate_disk(bundle.normalized_mesh)

    def record_pp_step(uv_after: np.ndarray, completed: int, pp_elapsed_ns: int) -> Optional[str]:
        """Synchronize one PP endpoint, record common metrics, and apply the stage stop."""

        sync_start = time.perf_counter_ns()
        bundle.problem.setVars(uv_after.ravel())
        elapsed_ns = pp_elapsed_ns + time.perf_counter_ns() - sync_start
        finite = recorder.record(
            bundle.problem, BenchmarkPhase.INITIAL_OPTIMIZATION, completed, elapsed_ns
        )
        if not finite:
            return "nonfinite_state"
        if completed >= max_iters:
            return "iteration_budget_exhausted"
        if _final_gradient_norm(bundle.problem) < run_spec.initial_optimization_grad_tol:
            return "gradient_tolerance"
        return None

    _, summary, _ = PP_utils.run_pp_true_area_from_uv(
        bundle.normalized_mesh,
        bundle.problem.getVars().reshape(-1, 2),
        execution_mode="benchmark",
        iteration_limit=max_iters,
        bound_distortion_K=250.0,
        post_step_cb=record_pp_step,
    )
    bundle.optimizer.options.hessianProjectionController.reset()
    reason = summary["termination_reason"]
    return InitialOptimizerResult(
        completed_iterations=summary["sum_iter"],
        benchmark_converged=reason == "gradient_tolerance",
        failed=reason == "nonfinite_state",
        termination_reason=reason,
        bundle=bundle,
    )


def register_builtin_initial_optimizers() -> InitialOptimizerRegistry:
    """Register the currently supported Phase-1 optimizer adapters."""

    registry = InitialOptimizerRegistry()
    registry.register(
        InitialOptimizerAdapter(
            key="newton",
            canonical_name="Newton",
            aliases=(),
            run_stage=run_newton_initial_optimizer,
        )
    )
    registry.register(
        InitialOptimizerAdapter(
            key="slim",
            canonical_name="SLIM",
            aliases=(),
            run_stage=run_slim_initial_optimizer,
        )
    )
    registry.register(
        InitialOptimizerAdapter(
            key="pp_truearea",
            canonical_name="PP_TrueArea",
            aliases=(),
            run_stage=run_pp_true_area_initial_optimizer,
        )
    )
    return registry


def run_complete_method(
    bundle: ProblemBundle,
    run_spec: RunSpec,
    writer: CsvIterationWriter,
    method_registry: MethodRegistry,
    initial_optimizer_registry: InitialOptimizerRegistry,
    initialization_elapsed_ns: int,
) -> RunOutcome:
    """Record initialization, then run optional Phase 1 and core Phase 2."""

    recorder = IterationRecorder(writer, run_spec)
    initialization_finite = recorder.record(
        bundle.problem,
        BenchmarkPhase.UV_INITIALIZATION,
        iteration_index=0,
        elapsed_ns=initialization_elapsed_ns,
    )
    benchmark.reset()
    if not initialization_finite:
        return RunOutcome(0, False, True, "nonfinite_initialization")

    completed = 0
    initial_stage = None
    if run_spec.initial_optimization_iters > 0:
        if run_spec.initial_optimizer_spec is None:
            raise ValueError("Positive Phase-1 cap requires an initial optimizer")
        initial_adapter = initial_optimizer_registry.get(
            run_spec.initial_optimizer_spec.adapter_key
        )
        initial_stage = initial_adapter.run_stage(
            bundle,
            run_spec,
            recorder,
            run_spec.initial_optimization_iters,
        )
        bundle = initial_stage.bundle
        completed += initial_stage.completed_iterations
        if initial_stage.failed:
            return RunOutcome(
                completed_iterations=completed,
                converged=False,
                failed=True,
                termination_reason=initial_stage.termination_reason,
            )

    if run_spec.method_spec is None:
        if initial_stage is None:
            return RunOutcome(0, False, False, "initial_optimization_disabled")
        return RunOutcome(
            completed_iterations=completed,
            converged=initial_stage.benchmark_converged,
            failed=False,
            termination_reason=initial_stage.termination_reason,
        )

    if (
        initial_stage is not None
        and _final_gradient_norm(bundle.problem) < GRADIENT_TOLERANCE
    ):
        return RunOutcome(completed, True, False, "gradient_tolerance")

    adapter = method_registry.get(run_spec.method_spec.adapter_key)

    if adapter.uses_initial_optimization and run_spec.initial_optimizer_spec is None:
        bundle.optimizer.options.hessianProjectionController.reset()

    remaining = run_spec.max_iters - completed
    if remaining <= 0:
        return RunOutcome(completed, False, False, "iteration_budget_exhausted")

    main_stage = adapter.run_main_stage(
        bundle,
        run_spec,
        recorder,
        completed,
        remaining,
    )
    completed += main_stage.completed_iterations
    return RunOutcome(
        completed_iterations=completed,
        converged=main_stage.converged,
        failed=main_stage.failed,
        termination_reason=main_stage.termination_reason,
    )


def set_thread_limit(thread_num: int) -> None:
    """Set MeshFEM's maximum TBB thread count for subsequent native work."""

    parallelism.set_max_num_tbb_threads(int(thread_num))


def validate_config(config: BenchmarkConfig) -> None:
    """Reject invalid global numeric, path, and list configuration."""

    if not config.models_dir.is_dir():
        raise ValueError(f"Models directory does not exist: {config.models_dir}")
    if config.output_csv.exists() and config.output_csv.is_dir():
        raise ValueError(f"Output CSV path is a directory: {config.output_csv}")
    if config.initial_only and config.requested_method_tokens:
        raise ValueError("--methods cannot be used with --initial-only")
    if config.initial_only and config.constant_speed_mode is not None:
        raise ValueError("--constant-speed cannot be used with --initial-only")
    if not config.initial_only and not config.requested_method_tokens:
        raise ValueError("At least one method is required")
    if not config.initializer_specs:
        raise ValueError("At least one initializer is required")
    if not config.initial_optimizer_specs:
        raise ValueError("--initial-optimizer must contain at least one name")
    if config.repeat_count < 1:
        raise ValueError("--repeat must be at least 1")
    if config.max_iters < 1:
        raise ValueError("--max-iters must be at least 1")
    if not config.initial_optimization_iters:
        raise ValueError(
            "--initial-optimization-iters must contain at least one integer"
        )
    if any(count < 0 for count in config.initial_optimization_iters):
        raise ValueError("--initial-optimization-iters values must be at least 0")
    if any(count > config.max_iters for count in config.initial_optimization_iters):
        raise ValueError(
            "--initial-optimization-iters values must not exceed --max-iters"
        )
    if len(set(config.initial_optimization_iters)) != len(
        config.initial_optimization_iters
    ):
        raise ValueError("--initial-optimization-iters must not contain duplicates")
    if not config.initial_optimization_grad_tols:
        raise ValueError("--initial-optimization-grad-tol needs at least one value")
    if any(
        not np.isfinite(tol) or tol <= 0
        for tol in config.initial_optimization_grad_tols
    ):
        raise ValueError("--initial-optimization-grad-tol values must be positive and finite")
    if len(set(config.initial_optimization_grad_tols)) != len(
        config.initial_optimization_grad_tols
    ):
        raise ValueError("--initial-optimization-grad-tol must not contain duplicates")
    if not config.thread_counts or any(thread < 1 for thread in config.thread_counts):
        raise ValueError("--threads must contain positive integers")
    if len(set(config.thread_counts)) != len(config.thread_counts):
        raise ValueError("--threads must not contain duplicates")


def initial_optimization_variants_for_method(
    config: BenchmarkConfig,
    method_spec: Optional[MethodSpec],
    method_registry: MethodRegistry,
) -> tuple[tuple[Optional[InitialOptimizerSpec], int, Optional[float]], ...]:
    """Return Phase-1 optimizer/limit/tolerance variants for one run."""

    if (
        method_spec is not None
        and not method_registry.get(method_spec.adapter_key).uses_initial_optimization
    ):
        return ((None, 0, None),)

    variants: list[tuple[Optional[InitialOptimizerSpec], int, Optional[float]]] = []
    if 0 in config.initial_optimization_iters:
        variants.append((None, 0, None))
    for optimizer_spec in config.initial_optimizer_specs:
        for count in config.initial_optimization_iters:
            if count > 0:
                variants.extend(
                    (optimizer_spec, count, tol)
                    for tol in config.initial_optimization_grad_tols
                )
    return tuple(variants)


def build_run_spec(
    config: BenchmarkConfig,
    model_path: Path,
    initializer_spec: InitializerSpec,
    method_spec: Optional[MethodSpec],
    thread_num: int,
    repeat_index: int,
    initial_optimizer_spec: Optional[InitialOptimizerSpec],
    initial_optimization_iters: int,
    initial_optimization_grad_tol: Optional[float],
) -> RunSpec:
    """Build one immutable complete-run specification from loop coordinates."""

    if initial_optimization_iters == 0 and initial_optimizer_spec is not None:
        raise ValueError("A zero Phase-1 cap requires initial_optimizer_spec=None")
    if initial_optimization_iters > 0 and initial_optimizer_spec is None:
        raise ValueError("A positive Phase-1 cap requires an initial optimizer")
    if initial_optimization_iters == 0 and initial_optimization_grad_tol is not None:
        raise ValueError("A zero Phase-1 cap requires no gradient tolerance")
    if initial_optimization_iters > 0 and initial_optimization_grad_tol is None:
        raise ValueError("A positive Phase-1 cap requires a gradient tolerance")
    return RunSpec(
        run_id=uuid4(),
        model_path=model_path,
        model_name=os.path.relpath(model_path, start=_MODULE_DIR),
        initializer_spec=initializer_spec,
        method_spec=method_spec,
        initial_optimizer_spec=initial_optimizer_spec,
        initial_optimization_iters=initial_optimization_iters,
        thread_num=thread_num,
        repeat_index=repeat_index,
        max_iters=config.max_iters,
        initial_optimization_grad_tol=initial_optimization_grad_tol,
    )


def build_experiment_matrix(
    config: BenchmarkConfig,
    models: Sequence[Path],
    methods: Sequence[MethodSpec],
    method_registry: MethodRegistry,
) -> list[RunSpec]:
    """Build every complete run specification in benchmark execution order."""

    matrix: list[RunSpec] = []
    for model_path in models:
        for initializer_spec in config.initializer_specs:
            for thread_num in config.thread_counts:
                for repeat_index in range(1, config.repeat_count + 1):
                    for method_spec in ((None,) if config.initial_only else methods):
                        phase_one_variants = initial_optimization_variants_for_method(
                            config,
                            method_spec,
                            method_registry,
                        )
                        for optimizer_spec, initial_iters, grad_tol in phase_one_variants:
                            matrix.append(
                                build_run_spec(
                                    config,
                                    model_path,
                                    initializer_spec,
                                    method_spec,
                                    thread_num,
                                    repeat_index,
                                    optimizer_spec,
                                    initial_iters,
                                    grad_tol,
                                )
                            )
    return matrix


def format_method_variant(method_spec: Optional[MethodSpec]) -> str:
    """Format a method variant for concise progress output."""

    if method_spec is None:
        return "N/A"
    if method_spec.constant_speed is None:
        return method_spec.display_name
    return f"{method_spec.display_name}[constant_speed={method_spec.constant_speed}]"


def native_thread_environment() -> dict[str, str]:
    """Return relevant native thread-pool environment values for provenance output."""

    import os

    names = (
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    return {name: os.environ.get(name, "<unset>") for name in names}
