import argparse
import time
from pathlib import Path
from statistics import mean, stdev

from stable_baselines3 import SAC

from finetune_sac import StartPhaseTorcsWrapper
from src.torcs_env import TorcsRLEnv
from src.torcs_process import TorcsProcessConfig, TorcsProcessManager


FULL_LAP_DIST_METERS = 3608.45


def resolve_model_path(model_path: str) -> str:
    """Accept model path with or without .zip and return an existing path."""
    path = Path(model_path)
    if path.exists():
        return str(path)

    zipped = path.with_suffix(path.suffix + ".zip") if path.suffix else Path(str(path) + ".zip")
    if zipped.exists():
        return str(zipped)

    raise FileNotFoundError(
        f"Model not found. Tried: '{path}' and '{zipped}'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAC checkpoint with the same wrapper used in finetune_sac_start.py"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Checkpoint path with or without .zip",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")

    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--relaunch-every", type=int, default=5)
    parser.add_argument("--startup-steps", type=int, default=220)
    parser.add_argument("--max-steps", type=int, default=100_000)

    parser.add_argument("--torcs-command", type=str, default="wine wtorcs.exe")
    parser.add_argument("--torcs-dir", type=str, default="torcs")
    parser.add_argument("--autostart-script", type=str, default="gym_torcs/autostart.sh")

    return parser.parse_args()


def build_env(args: argparse.Namespace) -> StartPhaseTorcsWrapper:
    process_manager = TorcsProcessManager(
        autostart_script=args.autostart_script,
        config=TorcsProcessConfig(
            torcs_command=args.torcs_command,
            torcs_working_dir=args.torcs_dir,
            vision=args.vision,
        ),
    )

    base_env = TorcsRLEnv(
        process_manager=process_manager,
        port=args.port,
        vision=args.vision,
        relaunch_every=args.relaunch_every,
        max_steps=args.max_steps,
    )

    return StartPhaseTorcsWrapper(base_env, startup_steps=args.startup_steps)


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)

    print("Creating evaluation environment (startup wrapper)...")
    env = build_env(args)

    print(f"Loading model: {model_path}")
    model = SAC.load(model_path, env=env)

    lap_times = []
    completed = 0

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            done = False
            steps = 0
            total_reward = 0.0
            last_info = {}

            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                last_info = info

                if truncated:
                    done = True

            dist = float(last_info.get("dist_raced", 0.0))
            last_lap = float(last_info.get("last_lap_time", 0.0))
            cur_lap = float(last_info.get("cur_lap_time", 0.0))

            lap_completed = last_lap > 0.0 or dist > FULL_LAP_DIST_METERS
            lap_time_to_use = last_lap if last_lap > 0.0 else cur_lap

            if lap_completed and lap_time_to_use > 0.0:
                completed += 1
                lap_times.append(lap_time_to_use)
                print(
                    f"Episode {ep}: COMPLETE | lap={lap_time_to_use:.3f}s | "
                    f"dist={dist:.1f}m | steps={steps} | reward={total_reward:.2f}"
                )
                
                print("Lap completed!")
                time.sleep(7) 
                
            else:
                print(
                    f"Episode {ep}: INCOMPLETE | dist={dist:.1f}m | "
                    f"time={cur_lap:.3f}s | steps={steps} | reward={total_reward:.2f}"
                )

    finally:
        env.close()

    print("\n===== Summary =====")
    print(f"Completed laps: {completed}/{args.episodes}")

    if lap_times:
        best = min(lap_times)
        avg = mean(lap_times)
        spread = stdev(lap_times) if len(lap_times) > 1 else 0.0
        print(f"Best lap: {best:.3f}s")
        print(f"Mean lap: {avg:.3f}s")
        print(f"Std dev: {spread:.3f}s")
        print("Lap list: " + ", ".join(f"{t:.3f}" for t in lap_times))
    else:
        print("No completed laps in this run.")


if __name__ == "__main__":
    main()