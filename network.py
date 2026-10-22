"""A small, inspectable NumPy MLP: no autograd or machine-learning framework."""

from dataclasses import asdict, dataclass, fields
import math

import numpy as np


@dataclass(frozen=True)
class Config:
    """Bounded settings keep experiments reproducible and requests inexpensive."""

    dataset: str = "xor"
    activation: str = "tanh"
    optimizer: str = "adam"
    samples: int = 400
    hidden: int = 12
    epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 0.03
    seed: int = 7

    @classmethod
    def parse(cls, payload):
        """Reject unknown fields and bool-as-int surprises instead of guessing."""
        if not isinstance(payload, dict):
            raise ValueError("Settings must be a JSON object.")
        known = {field.name for field in fields(cls)}
        if set(payload) - known:
            raise ValueError("Unknown settings: " + ", ".join(sorted(set(payload) - known)))
        config = cls(**payload)
        for name, choices in (
            ("dataset", ("xor", "circle")),
            ("activation", ("tanh", "relu")),
            ("optimizer", ("sgd", "adam")),
        ):
            if getattr(config, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}.")
        for name, low, high in (
            ("samples", 80, 800), ("hidden", 2, 32), ("epochs", 1, 1200),
            ("batch_size", 8, 128), ("seed", 0, 2**32 - 1),
        ):
            value = getattr(config, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer from {low} to {high}.")
        rate = config.learning_rate
        if type(rate) not in (int, float) or not 0 < rate <= 1 or not math.isfinite(rate):
            raise ValueError("learning_rate must be finite and greater than 0, at most 1.")
        return config


def sigmoid(z):
    """exp(-abs(z)) avoids overflow for either sign of the logit."""
    small = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + small), small / (1 + small))


def validate_xy(x, y=None):
    """Public numeric operations reject malformed, empty, or nonfinite arrays."""
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 2 or len(x) == 0:
        raise ValueError("X must be a nonempty NumPy array of shape (n, 2).")
    if not np.issubdtype(x.dtype, np.number) or not np.isrealobj(x) or not np.isfinite(x).all():
        raise ValueError("X must contain finite real numbers.")
    if y is not None:
        if not isinstance(y, np.ndarray) or y.shape != (len(x), 1):
            raise ValueError("y must have shape (n, 1).")
        if not np.issubdtype(y.dtype, np.number) or not np.isrealobj(y) or not np.isin(y, [0, 1]).all():
            raise ValueError("y must contain only binary labels 0 and 1.")


class MLP:
    """Two inputs → nonlinear hidden layer → one binary probability."""

    def __init__(self, hidden=12, activation="tanh", seed=7):
        if type(hidden) is not int or not 2 <= hidden <= 32:
            raise ValueError("hidden must be an integer from 2 to 32.")
        if activation not in ("tanh", "relu"):
            raise ValueError("activation must be tanh or relu.")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer from 0 to 4294967295.")
        self.activation = activation
        rng = np.random.default_rng(seed)
        # Xavier scaling for tanh, He scaling for ReLU; biases begin at zero.
        scale = np.sqrt((2 if activation == "relu" else 1) / 2)
        self.params = {
            "W1": rng.normal(size=(2, hidden)) * scale,
            "b1": np.zeros((1, hidden)),
            "W2": rng.normal(size=(hidden, 1)) / np.sqrt(hidden),
            "b2": np.zeros((1, 1)),
        }

    def forward(self, x):
        validate_xy(x)
        z1 = x @ self.params["W1"] + self.params["b1"]
        a1 = np.tanh(z1) if self.activation == "tanh" else np.maximum(0, z1)
        logits = a1 @ self.params["W2"] + self.params["b2"]
        return sigmoid(logits), (z1, a1, logits)

    def loss_and_gradients(self, x, y):
        """Differentiate mean BCE by the chain rule, including the 1/n factor."""
        validate_xy(x, y)
        probabilities, (z1, a1, logits) = self.forward(x)
        # log(1 + exp(logit)) - y*logit is stable even for extreme logits.
        loss = np.mean(np.logaddexp(0, logits) - y * logits)
        dz2 = (probabilities - y) / len(x)
        dw2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ self.params["W2"].T
        derivative = 1 - a1**2 if self.activation == "tanh" else (z1 > 0)
        dz1 = da1 * derivative
        gradients = {
            "W1": x.T @ dz1, "b1": dz1.sum(axis=0, keepdims=True),
            "W2": dw2, "b2": db2,
        }
        return float(loss), gradients

    def metrics(self, x, y):
        loss, _ = self.loss_and_gradients(x, y)
        probabilities, _ = self.forward(x)
        return {"loss": loss, "accuracy": float(np.mean((probabilities >= 0.5) == y))}


class Optimizer:
    """SGD or Adam, with independent state for each parameter tensor."""

    def __init__(self, name, learning_rate):
        if name not in ("sgd", "adam"):
            raise ValueError("optimizer must be sgd or adam.")
        if type(learning_rate) not in (int, float) or not 0 < learning_rate <= 1 or not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite and in (0, 1].")
        self.name, self.learning_rate = name, learning_rate
        self.t = 0
        self.m, self.v = {}, {}

    def step(self, params, gradients):
        # Validate every tensor before mutating anything, so failures are atomic.
        if set(params) != set(gradients):
            raise ValueError("Gradient and parameter keys must match.")
        for key, parameter in params.items():
            if parameter.shape != gradients[key].shape or not np.isfinite(gradients[key]).all():
                raise ValueError("Gradients must be finite and match parameter shapes.")
        self.t += 1
        for key, parameter in params.items():
            gradient = gradients[key]
            if self.name == "sgd":
                parameter -= self.learning_rate * gradient
            else:
                # Bias correction prevents the initial zero moments shrinking steps.
                self.m[key] = 0.9 * self.m.get(key, np.zeros_like(parameter)) + 0.1 * gradient
                self.v[key] = 0.999 * self.v.get(key, np.zeros_like(parameter)) + 0.001 * gradient**2
                m_hat = self.m[key] / (1 - 0.9**self.t)
                v_hat = self.v[key] / (1 - 0.999**self.t)
                parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)


def make_data(config):
    """Generate data locally and reserve 25% before fitting; no data downloads."""
    rng = np.random.default_rng(config.seed)
    x = rng.uniform(-1, 1, size=(config.samples, 2))
    # True XOR is 1 when the signs differ; circle is 1 outside radius sqrt(0.5).
    y = ((x[:, 0] * x[:, 1] < 0) if config.dataset == "xor"
         else (np.sum(x**2, axis=1) > 0.5)).astype(float).reshape(-1, 1)
    order = rng.permutation(len(x))
    split = int(0.75 * len(x))
    return x[order[:split]], y[order[:split]], x[order[split:]], y[order[split:]]


def train(payload):
    """Return measured histories, a probability grid, and one worked example."""
    config = Config.parse(payload)
    x, y, test_x, test_y = make_data(config)
    model = MLP(config.hidden, config.activation, config.seed)
    optimizer = Optimizer(config.optimizer, config.learning_rate)
    # Separate RNG streams make batch order independent of weight initialization.
    rng = np.random.default_rng(config.seed + 1)
    history = []

    def record(epoch):
        history.append({"epoch": epoch, "train": model.metrics(x, y),
                        "validation": model.metrics(test_x, test_y)})

    record(0)
    interval = max(1, math.ceil(config.epochs / 150))
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(x), config.batch_size):
            batch = order[start:start + config.batch_size]
            _, gradients = model.loss_and_gradients(x[batch], y[batch])
            optimizer.step(model.params, gradients)
        if epoch % interval == 0 or epoch == config.epochs:
            record(epoch)
    # Explicitly surface divergence instead of serializing NaN into browser JSON.
    if not all(np.isfinite(parameter).all() for parameter in model.params.values()):
        raise ValueError("Training diverged. Reduce the learning rate.")
    if not all(math.isfinite(item[split][metric]) for item in history
               for split in ("train", "validation") for metric in ("loss", "accuracy")):
        raise ValueError("Training produced nonfinite metrics. Reduce the learning rate.")

    # The tutorial uses one sample after training; it does NOT apply another update.
    one_x, one_y = x[:1], y[:1]
    probability, (z1, a1, logits) = model.forward(one_x)
    sample_loss, sample_gradients = model.loss_and_gradients(one_x, one_y)
    p, target = float(probability[0, 0]), float(one_y[0, 0])
    dz2 = p - target
    derivative = float(1 - a1[0, 0]**2) if config.activation == "tanh" else float(z1[0, 0] > 0)
    first_gradient = float(sample_gradients["W1"][0, 0])
    weight = float(model.params["W1"][0, 0])
    example = {
        "x": one_x[0].tolist(), "y": target, "z1": z1[0].tolist(),
        "a1": a1[0].tolist(), "logit": float(logits[0, 0]), "probability": p,
        "loss": sample_loss, "dz2": dz2, "activation_derivative": derivative,
        "w2_first": float(model.params["W2"][0, 0]),
        "gradient_w1_first": first_gradient, "weight_before": weight,
        "hypothetical_sgd_after": weight - config.learning_rate * first_gradient,
        "weights": {key: value.tolist() for key, value in model.params.items()},
    }
    # Row-major grid matches canvas pixels: first row represents y=-1.
    axis = np.linspace(-1, 1, 55)
    xx, yy = np.meshgrid(axis, axis)
    grid, _ = model.forward(np.column_stack([xx.ravel(), yy.ravel()]))
    return {
        "config": asdict(config), "history": history, "example": example,
        "grid": {"size": 55, "probabilities": grid.ravel().tolist()},
        "points": [
            {"x": row.tolist(), "y": int(label[0]), "split": split}
            for split, features, labels in (("train", x, y), ("validation", test_x, test_y))
            for row, label in zip(features, labels)
        ],
        "train_count": len(x), "validation_count": len(test_x),
        "updates": optimizer.t,
    }
