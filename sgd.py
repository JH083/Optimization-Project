"""Step 2: stochastic gradient descent baseline.

This module implements the first optimizer for the project comparison:
plain SGD on the shared L2-regularized logistic-regression objective.
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


SGDResult = OptimizationResult


def run_sgd(
    objective: LogisticRegressionObjective,
    theta0: np.ndarray | None = None,
    step_size: float = 0.01,
    epochs: int = 30,
    seed: int = 0,
    record_every_steps: int | None = None,
    val_objective: LogisticRegressionObjective | None = None,
) -> SGDResult:
    """Run SGD with one random sample per update.

    Args:
        objective: Shared logistic-regression objective.
        theta0: Initial parameter vector. Defaults to all zeros.
        step_size: Constant learning rate.
        epochs: Number of passes worth of stochastic updates.
        seed: Random seed used for sample selection.
        record_every_steps: Loss checkpoint frequency. Defaults to one epoch.
    """

    if step_size <= 0.0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    if record_every_steps is None:
        record_every_steps = objective.n_samples
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
    total_steps = epochs * objective.n_samples

    logger = CheckpointLogger(objective, val_objective=val_objective)
    logger.record(theta=theta, gradient_evaluations=0)

    grad_evals = 0
    for step in range(1, total_steps + 1):
        sample_index = int(rng.integers(objective.n_samples))
        gradient = objective.per_sample_gradient(theta, sample_index)
        theta -= step_size * gradient
        grad_evals += 1

        if step % record_every_steps == 0 or step == total_steps:
            logger.record(theta=theta, gradient_evaluations=grad_evals)

    return logger.build_result(
        method="SGD",
        theta=theta,
        step_size=step_size,
        epochs=epochs,
        seed=seed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 2 SGD smoke experiment.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda", dest="l2_reg", type=float, default=DEFAULT_L2_REG)
    parser.add_argument(
        "--record-every-steps",
        type=int,
        default=None,
        help="Loss checkpoint frequency. Defaults to one epoch.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    objective = make_objective(l2_reg=args.l2_reg)
    result = run_sgd(
        objective=objective,
        step_size=args.step_size,
        epochs=args.epochs,
        seed=args.seed,
        record_every_steps=args.record_every_steps,
    )

    print("Step 2 SGD smoke run")
    print(f"samples: {objective.n_samples}")
    print(f"features: {objective.n_features}")
    print(f"lambda: {objective.l2_reg:g}")
    print(f"step size: {result.step_size:g}")
    print(f"epochs: {result.epochs}")
    print(f"seed: {result.seed}")
    print(f"initial loss: {result.losses[0]:.8f}")
    print(f"final loss: {result.final_loss:.8f}")
    print(f"gradient evaluations: {result.total_gradient_evaluations}")
    print(f"runtime seconds: {result.total_runtime:.6f}")


if __name__ == "__main__":
    main()
