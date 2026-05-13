from dataclasses import dataclass
from typing import Iterator, Sequence


@dataclass(frozen=True)
class TorcsObservation:
    focus: float
    angle: float
    track: Sequence[float]
    trackPos: float
    speedX: float
    speedY: float
    speedZ: float
    wheelSpinVel: Sequence[float]
    rpm: float
    opponents: Sequence[float]
    vision: Sequence[float] | None = None

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(
            (
                "focus",
                "angle",
                "track",
                "trackPos",
                "speedX",
                "speedY",
                "speedZ",
                "wheelSpinVel",
                "rpm",
                "opponents",
                "vision",
            )
        )

    def keys(self) -> Iterator[str]:
        return iter(self)


@dataclass(frozen=True)
class TorcsAction:
    steering: float
    acceleration: float
    brake: float

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("steering", "acceleration", "brake"))

    def keys(self) -> Iterator[str]:
        return iter(self)
