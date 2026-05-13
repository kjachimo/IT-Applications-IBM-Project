from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class SteeringModel:
    """Small model-backed steering policy.

    The model uses a compact feature vector derived from the current TORCS state
    and predicts a steering correction in the range [-1, 1].
    """

    weights: list[float] = field(
        default_factory=lambda: [0.22, -1.05, -0.05, 0.28, 0.06, -0.18]
    )
    bias: float = 0.0
    model_path: Path | None = None

    def __post_init__(self) -> None:
        if self.model_path is None:
            self.model_path = Path(__file__).with_name("steering_model.json")
        self._load_if_available()

    def _load_if_available(self) -> None:
        if not self.model_path or not self.model_path.exists():
            return
        try:
            payload = json.loads(self.model_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return

        weights = payload.get("weights")
        bias = payload.get("bias")
        if isinstance(weights, list) and len(weights) == len(self.weights):
            self.weights = [float(value) for value in weights]
        if isinstance(bias, (int, float)):
            self.bias = float(bias)

    def save(self) -> None:
        if not self.model_path:
            return
        payload = {"weights": self.weights, "bias": self.bias}
        self.model_path.write_text(json.dumps(payload, indent=2))

    def _feature_vector(self, state: dict) -> list[float]:
        track = state.get("track") or []
        speed = float(state.get("speedX", 0.0)) / 300.0
        angle = float(state.get("angle", 0.0))
        track_pos = float(state.get("trackPos", 0.0))

        center = len(track) // 2 if track else 0
        front_clearance = float(track[center]) if track else 0.0
        left_clearance = self._weighted_side(track, 0, center, -1)
        right_clearance = self._weighted_side(track, center + 1, len(track), 1)
        balance = (right_clearance - left_clearance) / max(
            left_clearance + right_clearance, 1e-6
        )
        front_deviation = (front_clearance - 100.0) / 100.0
        curvature = self._curvature_hint(track)
        corner_pressure = self._corner_pressure(track)

        return [
            angle / math.pi,
            track_pos,
            speed,
            balance,
            front_deviation,
            curvature + corner_pressure,
        ]

    def predict(self, state: dict) -> float:
        features = self._feature_vector(state)
        steer = self.bias + sum(
            weight * value for weight, value in zip(self.weights, features)
        )
        turn_intensity = min(1.0, abs(features[-1]))
        return 0.55 * math.tanh(steer * (1.0 + 0.8 * turn_intensity))

    def turn_pressure(self, track: list[float]) -> float:
        return self._corner_pressure(track)

    def update(self, state: dict, target: float, learning_rate: float = 0.02) -> None:
        features = self._feature_vector(state)
        prediction = self.predict(state)
        error = float(target) - prediction
        scale = learning_rate * (1.0 - prediction * prediction)
        self.bias += scale * error
        self.weights = [weight + scale * error * value for weight, value in zip(self.weights, features)]

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        values = list(values)
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _weighted_side(track: list[float], start: int, stop: int, direction: int) -> float:
        values = track[start:stop]
        if not values:
            return 0.0
        weights = list(range(len(values), 0, -1))
        total_weight = sum(weights)
        return sum(value * weight for value, weight in zip(values, weights)) / total_weight

    @staticmethod
    def _curvature_hint(track: list[float]) -> float:
        if len(track) < 9:
            return 0.0
        center = len(track) // 2
        left = track[max(0, center - 4) : center]
        right = track[center + 1 : center + 5]
        left_mean = SteeringModel._weighted_side(left, 0, len(left), -1)
        right_mean = SteeringModel._weighted_side(right, 0, len(right), 1)
        return (right_mean - left_mean) / max(left_mean + right_mean, 1e-6)

    @staticmethod
    def _corner_pressure(track: list[float]) -> float:
        if len(track) < 9:
            return 0.0
        center = len(track) // 2
        left_front = SteeringModel._weighted_side(track[max(0, center - 4) : center], 0, max(0, center), -1)
        right_front = SteeringModel._weighted_side(track[center + 1 : center + 5], 0, min(4, len(track) - center - 1), 1)
        return (right_front - left_front) / max(left_front + right_front, 1e-6)
