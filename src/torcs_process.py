import os
import signal
import shutil
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TorcsProcessConfig:
    torcs_command: str = "torcs"
    torcs_working_dir: str = ""
    vision: bool = False
    nofuel: bool = True
    nodamage: bool = True
    nolaptime: bool = True
    launch_wait_sec: float = 0.8
    autostart_wait_sec: float = 0.8


class TorcsProcessManager:
    def __init__(
        self,
        autostart_script: str,
        config: Optional[TorcsProcessConfig] = None,
    ):
        self.config = config or TorcsProcessConfig()
        self.autostart_script = str(Path(autostart_script).resolve())
        self._torcs_proc = None

    def _launch_command(self):
        cmd = shlex.split(self.config.torcs_command)
        if self.config.nofuel:
            cmd.append("-nofuel")
        if self.config.nodamage:
            cmd.append("-nodamage")
        if self.config.nolaptime:
            cmd.append("-nolaptime")
        if self.config.vision:
            cmd.append("-vision")
        return cmd

    def _run_autostart(self):
        subprocess.run(["bash", self.autostart_script], check=False)

    def _pkill_torcs(self):
        # Prefer terminating the exact process group we launched to avoid
        # accidentally killing this Python trainer via command-line matches.
        if self._torcs_proc is not None and self._torcs_proc.poll() is None:
            try:
                os.killpg(self._torcs_proc.pid, signal.SIGTERM)
                time.sleep(0.3)
                if self._torcs_proc.poll() is None:
                    os.killpg(self._torcs_proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                self._torcs_proc = None

        # Fallback cleanup for native TORCS only. Avoid "-f torcs" patterns,
        # which may match user commands and terminate unrelated processes.
        subprocess.run(["pkill", "-x", "torcs"], check=False)

    def launch(self):
        self._pkill_torcs()
        launch_cwd = (
            str(Path(self.config.torcs_working_dir).resolve())
            if self.config.torcs_working_dir
            else None
        )
        self._torcs_proc = subprocess.Popen(
            self._launch_command(),
            cwd=launch_cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(self.config.launch_wait_sec)
        self._run_autostart()
        time.sleep(self.config.autostart_wait_sec)

    def hard_reset(self):
        self.launch()

    def stop(self):
        self._pkill_torcs()

    def check_requirements(self):
        missing = []
        command_tokens = shlex.split(self.config.torcs_command)
        launcher = command_tokens[0] if command_tokens else ""
        if not launcher:
            missing.append("torcs_command")
        elif shutil.which(launcher) is None:
            missing.append(launcher)

        if (
            command_tokens
            and command_tokens[0] == "wine"
            and len(command_tokens) > 1
            and command_tokens[1].endswith(".exe")
        ):
            exe_path = Path(command_tokens[1])
            if not exe_path.is_absolute() and self.config.torcs_working_dir:
                exe_path = Path(self.config.torcs_working_dir) / exe_path
            if not exe_path.exists():
                missing.append(str(exe_path))

        if shutil.which("xte") is None:
            missing.append("xte")
        if not Path(self.autostart_script).exists():
            missing.append(self.autostart_script)

        if (
            self.config.torcs_working_dir
            and not Path(self.config.torcs_working_dir).exists()
        ):
            missing.append(self.config.torcs_working_dir)

        return missing
