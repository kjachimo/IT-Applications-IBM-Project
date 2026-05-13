from pathlib import Path
import subprocess
import time


_TORCS_DIR = Path(__file__).resolve().parent.parent / "torcs"
_AUTOSTART_DIR = Path(__file__).resolve().parent


def launch_torcs(vision=False, exe="wtorcs.exe", torcs_dir=None, use_wine=True):
    cmd = [exe, "-nofuel", "-nodamage", "-nolaptime"]
    if use_wine:
        cmd.insert(0, "wine")
    if vision is True:
        cmd.append("-vision")
    return subprocess.Popen(cmd, cwd=str(torcs_dir or _TORCS_DIR))


def launch_autostart(autostart_dir=None):
    cmd = ["sh", "autostart.sh"]
    return subprocess.Popen(cmd, cwd=str(autostart_dir or _AUTOSTART_DIR))


def stop_torcs(process=None, kill_fallback=False):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    if kill_fallback:
        subprocess.run(["pkill", "torcs"], check=False)


def reset_torcs(
    vision=False,
    process=None,
    wait_after_stop=0.5,
    wait_after_launch=10.0,
    wait_after_autostart=1.0,
    exe="wtorcs.exe",
    torcs_dir=None,
    autostart_dir=None,
    kill_fallback=False,
    use_wine=True,
):
    stop_torcs(process, kill_fallback=kill_fallback)
    if wait_after_stop:
        time.sleep(wait_after_stop)
    process = launch_torcs(
        vision=vision,
        exe=exe,
        torcs_dir=torcs_dir,
        use_wine=use_wine,
    )
    if wait_after_launch:
        time.sleep(wait_after_launch)
    launch_autostart(autostart_dir=autostart_dir)
    if wait_after_autostart:
        time.sleep(wait_after_autostart)
    return process
