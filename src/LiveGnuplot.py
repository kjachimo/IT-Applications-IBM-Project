from pathlib import Path


class LiveGnuplot:
    def __init__(
        self,
        stream_path: str = "stream.xy",
        script_path: str = "live_xy_plot.gp",
        refresh_seconds: float = 0.2,
        reset_stream_file: bool = True,
        title: str = "Live XY Stream",
    ) -> None:
        self.stream_path = Path(stream_path)
        self.script_path = Path(script_path)
        self.refresh_seconds = refresh_seconds
        self.reset_stream_file = reset_stream_file
        self.title = title

        self.stream_path.parent.mkdir(parents=True, exist_ok=True)
        if self.reset_stream_file:
            self.stream_path.write_text("", encoding="utf-8")

        self.print_launch_command()

    def print_launch_command(self) -> None:
        cmd = (
            "gnuplot -e "
            f"\"datafile='{self.stream_path}'; refresh={self.refresh_seconds}; plottitle='{self.title}'\" "
            f"{self.script_path}"
        )
        print("[mapper] Launch gnuplot manually in another terminal:")
        print(f"[mapper] {cmd}")

    def append(self, x: float, y: float) -> None:
        with self.stream_path.open("a", encoding="utf-8") as f:
            f.write(f"{x:.6f} {y:.6f}\n")

    def close(self) -> None:
        return
