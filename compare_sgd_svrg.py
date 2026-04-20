"""SGD vs SVRG experiments: main comparison and learning-rate sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from data_objective import DEFAULT_L2_REG, make_objective
    from optimizer_common import (
        OptimizationResult,
        parse_step_sizes,
        plot_loss_vs_gradient_evaluations,
        plot_loss_vs_runtime,
        plot_sweep_final_losses,
        print_summary,
        write_sweep_summary_csv,
        write_trace_csv,
    )
    from sgd import SGDResult, run_sgd
    from svrg import SVRGResult, run_svrg
    from saga import SAGAResult, run_saga
except ModuleNotFoundError:
    from .data_objective import DEFAULT_L2_REG, make_objective
    from .optimizer_common import (
        OptimizationResult,
        parse_step_sizes,
        plot_loss_vs_gradient_evaluations,
        plot_loss_vs_runtime,
        plot_sweep_final_losses,
        print_summary,
        write_sweep_summary_csv,
        write_trace_csv,
    )
    from .sgd import SGDResult, run_sgd
    from .svrg import SVRGResult, run_svrg
    from .saga import SAGAResult, run_saga


LOCKED_SGD_STEP_SIZE = 0.03
LOCKED_SVRG_STEP_SIZE = 0.1
LOCKED_SAGA_STEP_SIZE = 0.1
DEFAULT_SEEDS = [0, 1, 2]


def run_comparison(
    epochs: int = 30,
    sgd_step_size: float = LOCKED_SGD_STEP_SIZE,
    svrg_step_size: float = LOCKED_SVRG_STEP_SIZE,
    saga_step_size: float = LOCKED_SAGA_STEP_SIZE,
    seed: int = 0,
    l2_reg: float = DEFAULT_L2_REG,
    output_dir: Path = Path("Project/outputs"),
) -> tuple[SGDResult, SVRGResult, SAGAResult]:
    """Run SGD, SVRG, and SAGA from the same initial theta and save logs/plots."""

    objective = make_objective(l2_reg=l2_reg)
    theta0 = objective.initial_theta()

    sgd_result = run_sgd(
        objective=objective,
        theta0=theta0,
        step_size=sgd_step_size,
        epochs=epochs,
        seed=seed,
    )
    svrg_result = run_svrg(
        objective=objective,
        theta0=theta0,
        step_size=svrg_step_size,
        epochs=epochs,
        inner_loop_steps=objective.n_samples,
        seed=seed,
    )
    saga_result = run_saga(
        objective=objective,
        theta0=theta0,
        step_size=saga_step_size,
        epochs=epochs,
        seed=seed,
    )

    results: dict[str, OptimizationResult] = {
        "SGD": sgd_result,
        "SVRG": svrg_result,
        "SAGA": saga_result,
    }
    write_trace_csv(output_dir / "sgd_vs_svrg_trace.csv", results)
    plot_loss_vs_gradient_evaluations(
        output_dir / "sgd_vs_svrg_loss_vs_grad_evals.png", results
    )
    plot_loss_vs_runtime(output_dir / "sgd_vs_svrg_loss_vs_runtime.png", results)
    print("SGD vs SVRG vs SAGA comparison")
    print_summary(results)
    print(f"trace: {output_dir / 'sgd_vs_svrg_trace.csv'}")
    print(f"plot: {output_dir / 'sgd_vs_svrg_loss_vs_grad_evals.png'}")
    print(f"runtime plot: {output_dir / 'sgd_vs_svrg_loss_vs_runtime.png'}")

    return sgd_result, svrg_result, saga_result


def run_seed_replicates(
    epochs: int = 30,
    seeds: list[int] | None = None,
    sgd_step_size: float = LOCKED_SGD_STEP_SIZE,
    svrg_step_size: float = LOCKED_SVRG_STEP_SIZE,
    saga_step_size: float = LOCKED_SAGA_STEP_SIZE,
    l2_reg: float = DEFAULT_L2_REG,
    output_dir: Path = Path("Project/outputs"),
) -> dict[str, dict[str, float]]:
    """Run the locked pipeline across seeds and summarize final metrics."""

    if seeds is None:
        seeds = DEFAULT_SEEDS

    objective = make_objective(l2_reg=l2_reg)
    theta0 = objective.initial_theta()
    results: list[OptimizationResult] = []

    for seed in seeds:
        results.append(
            run_sgd(
                objective=objective,
                theta0=theta0,
                step_size=sgd_step_size,
                epochs=epochs,
                seed=seed,
            )
        )
        results.append(
            run_svrg(
                objective=objective,
                theta0=theta0,
                step_size=svrg_step_size,
                epochs=epochs,
                inner_loop_steps=objective.n_samples,
                seed=seed,
            )
        )
        results.append(
            run_saga(
                objective=objective,
                theta0=theta0,
                step_size=saga_step_size,
                epochs=epochs,
                seed=seed,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = output_dir / "locked_seed_runs.csv"
    aggregate_path = output_dir / "locked_seed_summary.csv"

    _write_seed_runs_csv(per_seed_path, results)
    summary = _write_seed_summary_csv(aggregate_path, results, seeds)

    print("Locked 3-seed summary")
    for method, metrics in summary.items():
        print(
            f"{method}: mean final loss={metrics['mean_final_loss']:.8f}, "
            f"std={metrics['std_final_loss']:.8f}, "
            f"mean runtime={metrics['mean_runtime_seconds']:.6f}s"
        )
    print(f"per-seed runs: {per_seed_path}")
    print(f"seed summary: {aggregate_path}")

    return summary


def run_lr_sweeps(
    epochs: int = 30,
    sgd_step_sizes: list[float] | None = None,
    svrg_step_sizes: list[float] | None = None,
    saga_step_sizes: list[float] | None = None,
    seed: int = 0,
    l2_reg: float = DEFAULT_L2_REG,
    output_dir: Path = Path("Project/outputs"),
) -> list[OptimizationResult]:
    """Run learning-rate sweeps and write one summary row per run."""

    if sgd_step_sizes is None:
        sgd_step_sizes = [0.001, 0.003, 0.01, 0.03, 0.1]
    if svrg_step_sizes is None:
        svrg_step_sizes = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
    if saga_step_sizes is None:
        saga_step_sizes = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2]

    objective = make_objective(l2_reg=l2_reg)
    theta0 = objective.initial_theta()
    results: list[OptimizationResult] = []

    for step_size in sgd_step_sizes:
        results.append(
            run_sgd(
                objective=objective,
                theta0=theta0,
                step_size=step_size,
                epochs=epochs,
                seed=seed,
            )
        )

    for step_size in svrg_step_sizes:
        results.append(
            run_svrg(
                objective=objective,
                theta0=theta0,
                step_size=step_size,
                epochs=epochs,
                inner_loop_steps=objective.n_samples,
                seed=seed,
            )
        )

    for step_size in saga_step_sizes:
        results.append(
            run_saga(
                objective=objective,
                theta0=theta0,
                step_size=step_size,
                epochs=epochs,
                seed=seed,
            )
        )

    write_sweep_summary_csv(output_dir / "lr_sweep_summary.csv", results)
    plot_sweep_final_losses(output_dir / "lr_sweep_final_loss.png", results)
    print("Learning-rate sweep")
    _print_best_by_method(results)
    print(f"sweep summary: {output_dir / 'lr_sweep_summary.csv'}")
    print(f"sweep plot: {output_dir / 'lr_sweep_final_loss.png'}")
    return results


def _write_seed_runs_csv(output_path: Path, results: list[OptimizationResult]) -> None:
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "method",
                "seed",
                "step_size",
                "epochs",
                "final_loss",
                "runtime_seconds",
                "gradient_evaluations",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "method": result.method,
                    "seed": result.seed,
                    "step_size": f"{result.step_size:g}",
                    "epochs": result.epochs,
                    "final_loss": f"{result.final_loss:.10f}",
                    "runtime_seconds": f"{result.total_runtime:.10f}",
                    "gradient_evaluations": result.total_gradient_evaluations,
                }
            )


def _write_seed_summary_csv(
    output_path: Path,
    results: list[OptimizationResult],
    seeds: list[int],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "method",
                "seeds",
                "step_size",
                "epochs",
                "mean_final_loss",
                "std_final_loss",
                "mean_runtime_seconds",
                "std_runtime_seconds",
                "mean_gradient_evaluations",
            ],
        )
        writer.writeheader()

        for method in sorted({result.method for result in results}):
            method_results = [result for result in results if result.method == method]
            final_losses = np.asarray(
                [result.final_loss for result in method_results], dtype=float
            )
            runtimes = np.asarray(
                [result.total_runtime for result in method_results], dtype=float
            )
            grad_evals = np.asarray(
                [result.total_gradient_evaluations for result in method_results],
                dtype=float,
            )
            std_ddof = 1 if len(method_results) > 1 else 0
            metrics = {
                "mean_final_loss": float(np.mean(final_losses)),
                "std_final_loss": float(np.std(final_losses, ddof=std_ddof)),
                "mean_runtime_seconds": float(np.mean(runtimes)),
                "std_runtime_seconds": float(np.std(runtimes, ddof=std_ddof)),
                "mean_gradient_evaluations": float(np.mean(grad_evals)),
            }
            summary[method] = metrics
            writer.writerow(
                {
                    "method": method,
                    "seeds": " ".join(str(seed) for seed in seeds),
                    "step_size": f"{method_results[0].step_size:g}",
                    "epochs": method_results[0].epochs,
                    "mean_final_loss": f"{metrics['mean_final_loss']:.10f}",
                    "std_final_loss": f"{metrics['std_final_loss']:.10f}",
                    "mean_runtime_seconds": f"{metrics['mean_runtime_seconds']:.10f}",
                    "std_runtime_seconds": f"{metrics['std_runtime_seconds']:.10f}",
                    "mean_gradient_evaluations": (
                        f"{metrics['mean_gradient_evaluations']:.1f}"
                    ),
                }
            )

    return summary


def _print_best_by_method(results: list[OptimizationResult]) -> None:
    methods = sorted({result.method for result in results})
    for method in methods:
        method_results = [result for result in results if result.method == method]
        best = min(method_results, key=lambda result: result.final_loss)
        print(
            f"{method}: best step size={best.step_size:g}, "
            f"final loss={best.final_loss:.8f}, "
            f"runtime={best.total_runtime:.6f}s"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SGD and SVRG.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sgd-step-size", type=float, default=LOCKED_SGD_STEP_SIZE)
    parser.add_argument("--svrg-step-size", type=float, default=LOCKED_SVRG_STEP_SIZE)
    parser.add_argument("--saga-step-size", type=float, default=LOCKED_SAGA_STEP_SIZE)
    parser.add_argument(
        "--sgd-sweep",
        type=str,
        default="0.001,0.003,0.01,0.03,0.1",
        help="Comma-separated SGD learning rates.",
    )
    parser.add_argument(
        "--svrg-sweep",
        type=str,
        default="0.005,0.01,0.03,0.05,0.1,0.2",
        help="Comma-separated SVRG learning rates.",
    )
    parser.add_argument(
        "--saga-sweep",
        type=str,
        default="0.005,0.01,0.03,0.05,0.1,0.2",
        help="Comma-separated SAGA learning rates.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds for locked replicate runs.",
    )
    parser.add_argument("--lambda", dest="l2_reg", type=float, default=DEFAULT_L2_REG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Project/outputs"),
        help="Directory for CSV logs and plots.",
    )
    parser.add_argument(
        "--skip-sweeps",
        action="store_true",
        help="Only run the main SGD/SVRG comparison.",
    )
    parser.add_argument(
        "--run-seeds",
        action="store_true",
        help="Run locked seed replicates and aggregate final metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_comparison(
        epochs=args.epochs,
        sgd_step_size=args.sgd_step_size,
        svrg_step_size=args.svrg_step_size,
        saga_step_size=args.saga_step_size,
        seed=args.seed,
        l2_reg=args.l2_reg,
        output_dir=args.output_dir,
    )
    if not args.skip_sweeps:
        run_lr_sweeps(
            epochs=args.epochs,
            sgd_step_sizes=parse_step_sizes(args.sgd_sweep),
            svrg_step_sizes=parse_step_sizes(args.svrg_sweep),
            saga_step_sizes=parse_step_sizes(args.saga_sweep),
            seed=args.seed,
            l2_reg=args.l2_reg,
            output_dir=args.output_dir,
        )
    if args.run_seeds:
        run_seed_replicates(
            epochs=args.epochs,
            seeds=_parse_seeds(args.seeds),
            sgd_step_size=args.sgd_step_size,
            svrg_step_size=args.svrg_step_size,
            saga_step_size=args.saga_step_size,
            l2_reg=args.l2_reg,
            output_dir=args.output_dir,
        )


def _parse_seeds(raw_seeds: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw_seeds.split(",")]
    if not seeds:
        raise ValueError("seeds must be a comma-separated list of integers")
    return seeds


if __name__ == "__main__":
    main()
