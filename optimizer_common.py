"""Shared optimizer result and experiment utilities.

SGD, SVRG, and SAGA should all return ``OptimizationResult``.  Keeping one
result shape makes plotting, logging, and tradeoff tables independent of each
optimizer's internal update rule.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from data_objective import LogisticRegressionObjective
except ModuleNotFoundError:
    from .data_objective import LogisticRegressionObjective


@dataclass(frozen=True)
class OptimizationResult:
    """Common output contract for all optimizers."""

    method: str
    theta: np.ndarray
    losses: np.ndarray
    gradient_evaluations: np.ndarray
    runtimes: np.ndarray
    step_size: float
    epochs: int
    seed: int
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def final_loss(self) -> float:
        return float(self.losses[-1])

    @property
    def total_gradient_evaluations(self) -> int:
        return int(self.gradient_evaluations[-1])

    @property
    def total_runtime(self) -> float:
        return float(self.runtimes[-1])


class CheckpointLogger:
    """Collect loss/runtime/gradient-evaluation checkpoints."""

    def __init__(self, objective: LogisticRegressionObjective) -> None:
        self.objective = objective
        self.start_time = time.perf_counter()
        self.losses: list[float] = []
        self.gradient_evaluations: list[int] = []
        self.runtimes: list[float] = []

    def record(self, theta: np.ndarray, gradient_evaluations: int) -> None:
        self.losses.append(self.objective.objective(theta))
        self.gradient_evaluations.append(gradient_evaluations)
        self.runtimes.append(time.perf_counter() - self.start_time)

    def build_result(
        self,
        method: str,
        theta: np.ndarray,
        step_size: float,
        epochs: int,
        seed: int,
        metadata: dict[str, float | int | str] | None = None,
    ) -> OptimizationResult:
        return OptimizationResult(
            method=method,
            theta=theta,
            losses=np.asarray(self.losses, dtype=float),
            gradient_evaluations=np.asarray(self.gradient_evaluations, dtype=int),
            runtimes=np.asarray(self.runtimes, dtype=float),
            step_size=step_size,
            epochs=epochs,
            seed=seed,
            metadata={} if metadata is None else metadata,
        )


def write_trace_csv(
    output_path: Path,
    results: dict[str, OptimizationResult],
) -> None:
    """Write checkpoint logs for one or more optimizer runs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "method",
                "checkpoint",
                "gradient_evaluations",
                "runtime_seconds",
                "loss",
                "step_size",
                "epochs",
                "seed",
                "metadata",
            ],
        )
        writer.writeheader()

        for fallback_method, result in results.items():
            method = result.method or fallback_method
            metadata_json = json.dumps(result.metadata, sort_keys=True)
            for checkpoint, (grad_evals, runtime, loss) in enumerate(
                zip(result.gradient_evaluations, result.runtimes, result.losses)
            ):
                writer.writerow(
                    {
                        "method": method,
                        "checkpoint": checkpoint,
                        "gradient_evaluations": int(grad_evals),
                        "runtime_seconds": f"{float(runtime):.10f}",
                        "loss": f"{float(loss):.10f}",
                        "step_size": f"{result.step_size:g}",
                        "epochs": result.epochs,
                        "seed": result.seed,
                        "metadata": metadata_json,
                    }
                )


def write_sweep_summary_csv(
    output_path: Path,
    results: list[OptimizationResult],
) -> None:
    """Write one summary row per learning-rate sweep run."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "method",
                "step_size",
                "epochs",
                "seed",
                "final_loss",
                "gradient_evaluations",
                "runtime_seconds",
                "metadata",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "method": result.method,
                    "step_size": f"{result.step_size:g}",
                    "epochs": result.epochs,
                    "seed": result.seed,
                    "final_loss": f"{result.final_loss:.10f}",
                    "gradient_evaluations": result.total_gradient_evaluations,
                    "runtime_seconds": f"{result.total_runtime:.10f}",
                    "metadata": json.dumps(result.metadata, sort_keys=True),
                }
            )


def plot_loss_vs_gradient_evaluations(
    output_path: Path,
    results: dict[str, OptimizationResult],
) -> None:
    """Save objective value vs gradient evaluations."""

    _plot_metric(
        output_path=output_path,
        results=results,
        x_attr="gradient_evaluations",
        x_label="Gradient evaluations",
        title="Objective vs Gradient Evaluations",
    )


def plot_loss_vs_runtime(
    output_path: Path,
    results: dict[str, OptimizationResult],
) -> None:
    """Save objective value vs wall-clock runtime."""

    _plot_metric(
        output_path=output_path,
        results=results,
        x_attr="runtimes",
        x_label="Runtime seconds",
        title="Objective vs Runtime",
    )


def plot_sweep_final_losses(
    output_path: Path,
    results: list[OptimizationResult],
) -> None:
    """Save final objective value vs learning rate for sweep runs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for method in sorted({result.method for result in results}):
        method_results = sorted(
            [result for result in results if result.method == method],
            key=lambda result: result.step_size,
        )
        ax.semilogx(
            [result.step_size for result in method_results],
            [result.final_loss for result in method_results],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=method,
        )

    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Final objective value")
    ax.set_title("Learning-Rate Sensitivity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_summary(results: dict[str, OptimizationResult]) -> None:
    for method, result in results.items():
        print(
            f"{method}: final loss={result.final_loss:.8f}, "
            f"grad evals={result.total_gradient_evaluations}, "
            f"runtime={result.total_runtime:.6f}s, "
            f"step size={result.step_size:g}"
        )


def parse_step_sizes(raw_step_sizes: str) -> list[float]:
    step_sizes = [float(value.strip()) for value in raw_step_sizes.split(",")]
    if not step_sizes or any(step_size <= 0.0 for step_size in step_sizes):
        raise ValueError("step sizes must be a comma-separated list of positives")
    return step_sizes


def _plot_metric(
    output_path: Path,
    results: dict[str, OptimizationResult],
    x_attr: str,
    x_label: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, result in results.items():
        ax.plot(
            getattr(result, x_attr),
            result.losses,
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            label=method,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Objective value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
