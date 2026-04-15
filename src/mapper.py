import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.LiveGnuplot import LiveGnuplot

STEER_GAIN = 30


class Mapper:
    def __init__(
        self,
        print_image: bool = False,
        image_path: str = "track.png",
        live_plot: bool = False,
        plotter: Optional[LiveGnuplot] = None,
    ) -> None:
        self.print_image = print_image
        self.image_path = image_path
        self.live_plot = live_plot
        self.track_points: List[Tuple[float, float]] = []
        self.telemetry_log: List[Dict[str, Any]] = []
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._last_time: Optional[float] = None
        self._frame_count = 0
        self.plotter = plotter
        if self.live_plot and self.plotter is None:
            self.plotter = LiveGnuplot()

    def update(self, telemetry, dt=None):
        self._frame_count += 1
        speed = float(telemetry.get("speed", telemetry.get("speedX", 0.0)))
        angle = float(telemetry.get("angle", 0.0)) * STEER_GAIN / math.pi
        t = telemetry.get("time")
        dt = float(telemetry.get("dt", 0.02))
        if t is not None:
            t = float(t)
            if self._last_time is not None:
                dt = max(0.0, t - self._last_time)
            self._last_time = t
        # Integrate heading and project speed to local XY map.
        self._heading += angle * dt
        self._x += speed * dt * math.cos(self._heading)
        self._y += speed * dt * math.sin(self._heading)

        pos = (self._x, self._y)
        self.track_points.append(pos)
        self.telemetry_log.append(dict(telemetry, x=self._x, y=self._y))
        if self.live_plot and self.plotter is not None and self._frame_count % 10 == 0:
            self.plotter.append(self._x, self._y)

        return pos

    def _print_track(self, show: bool = True, save_path: Optional[str] = None) -> None:
        if not self.track_points:
            print("(no track points)")
            return

        import matplotlib.pyplot as plt

        xs = [p[0] for p in self.track_points]
        ys = [p[1] for p in self.track_points]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(xs, ys, "-", linewidth=1.5)
        ax.scatter(xs[-1], ys[-1], s=30)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Learned Track")
        ax.grid(True, alpha=0.3)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def save_json(self, path: str = "track_data.json") -> None:
        if self.print_image:
            self._print_track(show=False, save_path=self.image_path)
        data = {
            "track_points": [{"x": x, "y": y} for x, y in self.track_points],
            "telemetry": self.telemetry_log,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def close(self) -> None:
        if self.plotter is not None:
            self.plotter.close()

    def __del__(self) -> None:
        self.close()
