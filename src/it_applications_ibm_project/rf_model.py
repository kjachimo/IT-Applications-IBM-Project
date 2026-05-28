from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None


class RandomForestSteeringModel:
    """Random Forest wrapper for predicting steering corrections.

    Usage:
    - Collect training samples as (state_dict, target_steer) pairs.
    - Call `fit(samples)` to train (requires scikit-learn).
    - Use `predict(state)` to get a steering prediction in [-1, 1].
    - Save/load with `save(path)` / `load(path)`.
    """

    def __init__(self, model_path: Path | str | None = None, n_estimators: int = 100):
        self.model_path = (
            Path(model_path)
            if model_path
            else Path(__file__).with_name("rf_steering.pkl")
        )
        self.n_estimators = n_estimators
        self._model: Any | None = None

    @staticmethod
    def _feature_vector(state: dict) -> list[float]:
        track = state.get("track") or []
        speed = float(state.get("speedX", 0.0)) / 300.0
        angle = float(state.get("angle", 0.0))
        track_pos = float(state.get("trackPos", 0.0))

        center = len(track) // 2 if track else 0
        front_clearance = float(track[center]) if track else 0.0

        left = track[:center]
        right = track[center + 1 :]
        left_mean = sum(left) / len(left) if left else 0.0
        right_mean = sum(right) / len(right) if right else 0.0
        balance = (right_mean - left_mean) / max(left_mean + right_mean, 1e-6)

        return [angle, track_pos, speed, balance, front_clearance]

    def fit(self, samples: list[tuple[dict, float]]):
        if RandomForestRegressor is None:
            raise RuntimeError(
                "scikit-learn is required to train RandomForestSteeringModel"
            )
        X = [self._feature_vector(s) for s, _ in samples]
        y = [float(t) for _, t in samples]
        self._model = RandomForestRegressor(n_estimators=self.n_estimators)
        self._model.fit(X, y)
        return self

    def predict(self, state: dict) -> float:
        if self._model is None:
            # fallback: simple heuristic similar to SteeringModel
            feats = self._feature_vector(state)
            # basic linear fallback
            steer = 0.0
            if feats:
                steer = (feats[0] / 3.14) * 0.5 + feats[2] * 0.3 - feats[3] * 0.2
            return max(-1.0, min(1.0, steer))
        feats = [self._feature_vector(state)]
        pred = float(self._model.predict(feats)[0])
        return max(-1.0, min(1.0, pred))

    def save(self, path: Path | str | None = None) -> None:
        target = Path(path) if path else self.model_path
        if self._model is None:
            # save empty metadata
            target.write_bytes(b"")
            return
        with target.open("wb") as fh:
            pickle.dump({"n_estimators": self.n_estimators, "model": self._model}, fh)

    def load(self, path: Path | str | None = None) -> None:
        target = Path(path) if path else self.model_path
        if not target.exists():
            return
        try:
            with target.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception:
            return
        self.n_estimators = int(payload.get("n_estimators", self.n_estimators))
        self._model = payload.get("model")

    def has_model(self) -> bool:
        return self._model is not None
