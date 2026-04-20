"""SAGA optimizer for L2-regularized logistic regression.

SAGA maintains a table of per-sample gradients and uses them to form a
variance-reduced update at each step:

    v = grad_f_j(theta) - table[j] + mean(table) + lambda * w

After the update, table[j] is replaced with the freshly computed gradient
and the running mean is updated in O(d) time.
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


SAGAResult = OptimizationResult


def run_saga(
    objective: LogisticRegressionObjective,
    theta0: np.ndarray | None = None,
    step_size: float = 0.05,
    epochs: int = 30,
    seed: int = 0,
    record_every_steps: int | None = None,
    val_objective: LogisticRegressionObjective | None = None,
) -> SAGAResult:
    """Run SAGA from a given (or default zero) starting point.

    The gradient table is initialized by computing per-sample gradients at
    theta0, costing n evaluations before the first update.  This is counted
    in gradient_evaluations so comparisons against SVRG remain fair.

    Args:
        objective: Shared logistic-regression objective.
        theta0: Initial parameter vector. Defaults to all zeros.
        step_size: Constant learning rate.
        epochs: Number of passes worth of stochastic updates (each pass = n steps).
        seed: Random seed used for sample selection.
        record_every_steps: Loss checkpoint frequency. Defaults to one epoch.
    """

    if step_size <= 0.0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    n = objective.n_samples
    d = objective.n_features + 1

    if record_every_steps is None:
        record_every_steps = n
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

    # Build gradient table without regularization so table entries stay valid
    # as theta moves.  Regularization is added fresh at each update.
    grad_table = np.array(
        [objective.per_sample_gradient(theta, i, include_regularization=False) for i in range(n)],
        dtype=float,
    )
    table_mean = grad_table.mean(axis=0)

    rng = np.random.default_rng(seed)
    total_steps = epochs * n

    logger = CheckpointLogger(objective, val_objective=val_objective)
    logger.record(theta=theta, gradient_evaluations=n)

    grad_evals = n
    for step in range(1, total_steps + 1):
        j = int(rng.integers(n))

        new_grad_j = objective.per_sample_gradient(theta, j, include_regularization=False)

        w, _ = objective.split_theta(theta)
        reg_grad = np.zeros(d, dtype=float)
        reg_grad[:-1] = objective.l2_reg * w

        variance_reduced = new_grad_j - grad_table[j] + table_mean + reg_grad

        # O(d) table-mean update avoids re-summing the full table each step.
        table_mean += (new_grad_j - grad_table[j]) / n
        grad_table[j] = new_grad_j

        theta -= step_size * variance_reduced
        grad_evals += 1

        if step % record_every_steps == 0 or step == total_steps:
            logger.record(theta=theta, gradient_evaluations=grad_evals)

    return logger.build_result(
        method="SAGA",
        theta=theta,
        step_size=step_size,
        epochs=epochs,
        seed=seed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAGA smoke experiment.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=0.05)
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
    result = run_saga(
        objective=objective,
        step_size=args.step_size,
        epochs=args.epochs,
        seed=args.seed,
        record_every_steps=args.record_every_steps,
    )

    print("SAGA smoke run")
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
