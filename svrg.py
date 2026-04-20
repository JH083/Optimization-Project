"""Step 3: stochastic variance-reduced gradient baseline.

SVRG uses a periodically refreshed snapshot parameter and full gradient to
reduce the variance of stochastic updates:

    grad f_i(theta) - grad f_i(snapshot) + full_grad(snapshot)

The implementation mirrors the SGD module's result format so later experiment
code can plot SGD, SVRG, and SAGA on the same axes.
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from data_objective import (
        DEFAULT_L2_REG,
        LogisticRegressionObjective,
        make_objective,
    )
    from optimizer_common import CheckpointLogger, OptimizationResult
except ModuleNotFoundError:
    from .data_objective import (
        DEFAULT_L2_REG,
        LogisticRegressionObjective,
        make_objective,
    )
    from .optimizer_common import CheckpointLogger, OptimizationResult


SVRGResult = OptimizationResult


def run_svrg(
    objective: LogisticRegressionObjective,
    theta0: np.ndarray | None = None,
    step_size: float = 0.05,
    epochs: int = 30,
    inner_loop_steps: int | None = None,
    seed: int = 0,
    record_every_steps: int | None = None,
    val_objective: LogisticRegressionObjective | None = None,
) -> SVRGResult:
    """Run SVRG with snapshot frequency approximately one epoch.

    Args:
        objective: Shared logistic-regression objective.
        theta0: Initial parameter vector. Defaults to all zeros.
        step_size: Constant learning rate.
        epochs: Number of snapshot/inner-loop rounds.
        inner_loop_steps: Updates per snapshot. Defaults to n samples.
        seed: Random seed used for sample selection.
        record_every_steps: Loss checkpoint frequency in inner updates. Defaults
            to one inner loop.
    """

    if step_size <= 0.0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    if inner_loop_steps is None:
        inner_loop_steps = objective.n_samples
    if inner_loop_steps <= 0:
        raise ValueError(f"inner_loop_steps must be positive, got {inner_loop_steps}")

    if record_every_steps is None:
        record_every_steps = inner_loop_steps
    if record_every_steps <= 0:
        raise ValueError(
            f"record_every_steps must be positive, got {record_every_steps}"
        )

    theta = (
        objective.initial_theta()
        if theta0 is None
        else np.asarray(theta0, dtype=float).copy()
    )
    objective.split_theta(theta)

    rng = np.random.default_rng(seed)
    total_steps = epochs * inner_loop_steps

    logger = CheckpointLogger(objective, val_objective=val_objective)
    logger.record(theta=theta, gradient_evaluations=0)

    grad_evals = 0
    update_steps = 0
    for _ in range(epochs):
        snapshot_theta = theta.copy()
        snapshot_full_gradient = objective.full_gradient(snapshot_theta)
        grad_evals += objective.n_samples

        for _ in range(inner_loop_steps):
            sample_index = int(rng.integers(objective.n_samples))
            current_gradient = objective.per_sample_gradient(theta, sample_index)
            snapshot_gradient = objective.per_sample_gradient(
                snapshot_theta, sample_index
            )
            variance_reduced_gradient = (
                current_gradient - snapshot_gradient + snapshot_full_gradient
            )

            theta -= step_size * variance_reduced_gradient
            grad_evals += 2
            update_steps += 1

            if update_steps % record_every_steps == 0 or update_steps == total_steps:
                logger.record(theta=theta, gradient_evaluations=grad_evals)

    return logger.build_result(
        method="SVRG",
        theta=theta,
        step_size=step_size,
        epochs=epochs,
        seed=seed,
        metadata={"inner_loop_steps": inner_loop_steps},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 3 SVRG smoke experiment.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument(
        "--inner-loop-steps",
        type=int,
        default=None,
        help="Updates per snapshot. Defaults to the number of samples.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda", dest="l2_reg", type=float, default=DEFAULT_L2_REG)
    parser.add_argument(
        "--record-every-steps",
        type=int,
        default=None,
        help="Loss checkpoint frequency in inner updates. Defaults to one inner loop.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    objective = make_objective(l2_reg=args.l2_reg)
    result = run_svrg(
        objective=objective,
        step_size=args.step_size,
        epochs=args.epochs,
        inner_loop_steps=args.inner_loop_steps,
        seed=args.seed,
        record_every_steps=args.record_every_steps,
    )

    print("Step 3 SVRG smoke run")
    print(f"samples: {objective.n_samples}")
    print(f"features: {objective.n_features}")
    print(f"lambda: {objective.l2_reg:g}")
    print(f"step size: {result.step_size:g}")
    print(f"epochs: {result.epochs}")
    print(f"inner loop steps: {result.metadata['inner_loop_steps']}")
    print(f"seed: {result.seed}")
    print(f"initial loss: {result.losses[0]:.8f}")
    print(f"final loss: {result.final_loss:.8f}")
    print(f"gradient evaluations: {result.total_gradient_evaluations}")
    print(f"runtime seconds: {result.total_runtime:.6f}")


if __name__ == "__main__":
    main()
