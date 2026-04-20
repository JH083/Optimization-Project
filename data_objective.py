"""Step 1: data loading and logistic-regression objective utilities.

The project compares SGD, SVRG, and SAGA on the Breast Cancer dataset for
L2-regularized binary logistic regression.  This module provides the shared
data preprocessing and objective/gradient routines that all optimizers should
call so their comparisons stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_L2_REG = 1e-3


@dataclass(frozen=True)
class BinaryClassificationData:
    """Standardized binary classification data with labels in {-1, +1}."""

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


@dataclass(frozen=True)
class LogisticRegressionObjective:
    """L2-regularized logistic-regression objective.

    Parameters are represented as a single vector theta = [w_1, ..., w_d, b].
    The intercept b is intentionally not regularized.
    """

    data: BinaryClassificationData
    l2_reg: float = DEFAULT_L2_REG

    @property
    def n_samples(self) -> int:
        return self.data.n_samples

    @property
    def n_features(self) -> int:
        return self.data.n_features

    def initial_theta(self, scale: float = 0.0, seed: int | None = None) -> np.ndarray:
        """Create an initial parameter vector.

        By default this returns all zeros.  Passing a positive scale returns a
        reproducible Gaussian initialization, useful for later sensitivity tests.
        """

        if scale == 0.0:
            return np.zeros(self.n_features + 1, dtype=float)

        rng = np.random.default_rng(seed)
        return rng.normal(loc=0.0, scale=scale, size=self.n_features + 1)

    def split_theta(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        theta = np.asarray(theta, dtype=float)
        expected_shape = (self.n_features + 1,)
        if theta.shape != expected_shape:
            raise ValueError(f"theta must have shape {expected_shape}, got {theta.shape}")
        return theta[:-1], float(theta[-1])

    def margins(self, theta: np.ndarray) -> np.ndarray:
        w, b = self.split_theta(theta)
        return self.data.y * (self.data.X @ w + b)

    def average_logistic_loss(self, theta: np.ndarray) -> float:
        """Return mean log(1 + exp(-y_i * (w^T x_i + b)))."""

        return float(np.mean(np.logaddexp(0.0, -self.margins(theta))))

    def regularization_loss(self, theta: np.ndarray) -> float:
        """Return lambda / 2 * ||w||_2^2, excluding the bias."""

        w, _ = self.split_theta(theta)
        return float(0.5 * self.l2_reg * np.dot(w, w))

    def objective(self, theta: np.ndarray) -> float:
        """Return the full objective value F(w, b)."""

        return self.average_logistic_loss(theta) + self.regularization_loss(theta)

    def per_sample_gradient(
        self,
        theta: np.ndarray,
        sample_index: int,
        include_regularization: bool = True,
    ) -> np.ndarray:
        """Return gradient of one finite-sum term with respect to theta.

        With include_regularization=True, this is the gradient of
        log(1 + exp(-y_i z_i)) + lambda / 2 * ||w||_2^2.  Averaging these
        gradients over all samples gives the full objective gradient exactly.
        """

        if sample_index < 0 or sample_index >= self.n_samples:
            raise IndexError(
                f"sample_index must be in [0, {self.n_samples}), got {sample_index}"
            )

        w, b = self.split_theta(theta)
        x_i = self.data.X[sample_index]
        y_i = self.data.y[sample_index]
        margin_i = y_i * (float(x_i @ w) + b)
        coeff = -y_i * float(_sigmoid_negative_margin(np.array([margin_i]))[0])

        grad_w = coeff * x_i
        if include_regularization:
            grad_w = grad_w + self.l2_reg * w

        return np.concatenate([grad_w, np.array([coeff])])

    def full_gradient(self, theta: np.ndarray) -> np.ndarray:
        """Return gradient of the full regularized objective."""

        w, _ = self.split_theta(theta)
        coeffs = -self.data.y * _sigmoid_negative_margin(self.margins(theta))

        grad_w = (self.data.X.T @ coeffs) / self.n_samples + self.l2_reg * w
        grad_b = float(np.mean(coeffs))
        return np.concatenate([grad_w, np.array([grad_b])])


def load_breast_cancer_data() -> BinaryClassificationData:
    """Load and standardize sklearn's Breast Cancer dataset.

    Labels are converted from sklearn's {0, 1} convention to {-1, +1}.
    Feature standardization is fit on the full dataset for this optimization
    study; later train/test splits should reuse this preprocessing choice
    consistently across all methods.
    """

    raw = load_breast_cancer()
    scaler = StandardScaler()
    X = scaler.fit_transform(raw.data.astype(float))
    y = np.where(raw.target == 1, 1.0, -1.0)

    return BinaryClassificationData(
        X=X,
        y=y,
        feature_names=tuple(raw.feature_names),
        target_names=tuple(raw.target_names),
    )


def load_breast_cancer_data_split(
    test_size: float = 0.2,
    seed: int = 0,
) -> tuple[BinaryClassificationData, BinaryClassificationData]:
    """Return stratified train/test splits with scaler fit on train only."""

    raw = load_breast_cancer()
    X_raw = raw.data.astype(float)
    y = np.where(raw.target == 1, 1.0, -1.0)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=seed, stratify=raw.target
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    feature_names = tuple(raw.feature_names)
    target_names = tuple(raw.target_names)
    train_data = BinaryClassificationData(
        X=X_train, y=y_train,
        feature_names=feature_names, target_names=target_names,
    )
    val_data = BinaryClassificationData(
        X=X_test, y=y_test,
        feature_names=feature_names, target_names=target_names,
    )
    return train_data, val_data


def make_objective(l2_reg: float = DEFAULT_L2_REG) -> LogisticRegressionObjective:
    """Convenience constructor for the project objective."""

    return LogisticRegressionObjective(data=load_breast_cancer_data(), l2_reg=l2_reg)


def gradient_check(
    objective: LogisticRegressionObjective,
    theta: np.ndarray,
    epsilon: float = 1e-6,
    coordinates: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    """Compare analytic full gradient with centered finite differences.

    The check samples a few coordinates so it stays cheap enough to run often.
    """

    rng = np.random.default_rng(seed)
    grad = objective.full_gradient(theta)
    n_coords = min(coordinates, theta.size)
    coord_ids = rng.choice(theta.size, size=n_coords, replace=False)

    max_abs_error = 0.0
    for coord_id in coord_ids:
        step = np.zeros_like(theta)
        step[coord_id] = epsilon
        numeric = (
            objective.objective(theta + step) - objective.objective(theta - step)
        ) / (2.0 * epsilon)
        max_abs_error = max(max_abs_error, abs(numeric - grad[coord_id]))

    per_sample_mean = np.mean(
        [objective.per_sample_gradient(theta, i) for i in range(objective.n_samples)],
        axis=0,
    )
    max_mean_error = float(np.max(np.abs(per_sample_mean - grad)))

    return {
        "max_finite_difference_error": float(max_abs_error),
        "max_per_sample_mean_error": max_mean_error,
    }


def _sigmoid_negative_margin(margins: np.ndarray) -> np.ndarray:
    """Compute 1 / (1 + exp(margin)) without overflow."""

    margins = np.asarray(margins, dtype=float)
    values = np.empty_like(margins)

    nonnegative = margins >= 0.0
    exp_neg = np.exp(-margins[nonnegative])
    values[nonnegative] = exp_neg / (1.0 + exp_neg)

    exp_pos = np.exp(margins[~nonnegative])
    values[~nonnegative] = 1.0 / (1.0 + exp_pos)

    return values


def main() -> None:
    objective = make_objective()
    theta0 = objective.initial_theta()
    checks = gradient_check(objective, theta0)

    class_balance = {
        "-1": int(np.sum(objective.data.y == -1.0)),
        "+1": int(np.sum(objective.data.y == 1.0)),
    }

    print("Step 1 data + objective smoke check")
    print(f"samples: {objective.n_samples}")
    print(f"features: {objective.n_features}")
    print(f"class balance: {class_balance}")
    print(f"lambda: {objective.l2_reg:g}")
    print(f"objective(theta0): {objective.objective(theta0):.8f}")
    print(f"gradient norm(theta0): {np.linalg.norm(objective.full_gradient(theta0)):.8f}")
    print(
        "max finite-difference error: "
        f"{checks['max_finite_difference_error']:.3e}"
    )
    print(
        "max per-sample mean error: "
        f"{checks['max_per_sample_mean_error']:.3e}"
    )


if __name__ == "__main__":
    main()
