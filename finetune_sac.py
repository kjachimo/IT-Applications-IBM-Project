import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from src.torcs_env import TorcsRLEnv
from src.torcs_process import TorcsProcessConfig, TorcsProcessManager


# Fixed driving constraints
LAUNCH_CONTROL_STEPS = 120
LAUNCH_MIN_THROTTLE = 1.0
LAUNCH_MIN_SPEED = 45.0

CORNER_ENTRY_DIST = 78.0
CORNER_MAX_SPEED = 120.0
STRAIGHT_MIN_DIST = 120.0


class StartPhaseTorcsWrapper(gym.Env):
    """
    Startup-focused wrapper.

    Keeps controls and reward close to the original successful training setup,
    then applies extra shaping only in the first startup_steps of each episode.
    Episodes end naturally via TorcsRLEnv (off-track/backward/stall) or by
    lap completion when lastLapTime becomes available.
    """

    def __init__(
        self,
        torcs_env: TorcsRLEnv,
        startup_steps: int = 220,
    ):
        super().__init__()
        self.env = torcs_env

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.state_dim,),
            dtype=np.float32,
        )

        self.startup_steps = startup_steps
        self.corner_entry_dist = CORNER_ENTRY_DIST
        self.corner_max_speed = CORNER_MAX_SPEED
        self.straight_min_dist = STRAIGHT_MIN_DIST
        self.launch_control_steps = LAUNCH_CONTROL_STEPS
        self.launch_min_throttle = LAUNCH_MIN_THROTTLE
        self.launch_min_speed = LAUNCH_MIN_SPEED
        self.episode_count = 0
        self.relaunch_every = self.env.relaunch_every
        self.prev_steer = 0.0
        self.prev_speed_x = 0.0
        self.current_speed_x = 0.0
        self.prev_forward_dist = 200.0
        self.ep_speed_sum = 0.0
        self.ep_steps = 0
        self.ep_max_speed = 0.0
        self.heavy_turn_speed_sum = 0.0
        self.heavy_turn_steps = 0
        self.medium_turn_speed_sum = 0.0
        self.medium_turn_steps = 0
        self.light_turn_speed_sum = 0.0
        self.light_turn_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        relaunch = self.episode_count == 1 or (self.episode_count - 1) % self.relaunch_every == 0

        while True:
            try:
                state = self.env.reset(relaunch=relaunch)
                self.prev_steer = 0.0
                self.prev_speed_x = 0.0
                self.current_speed_x = 0.0
                self.prev_forward_dist = 200.0
                self.ep_speed_sum = 0.0
                self.ep_steps = 0
                self.ep_max_speed = 0.0
                self.heavy_turn_speed_sum = 0.0
                self.heavy_turn_steps = 0
                self.medium_turn_speed_sum = 0.0
                self.medium_turn_steps = 0
                self.light_turn_speed_sum = 0.0
                self.light_turn_steps = 0
                break
            except Exception as exc:
                print(f"[StartPhaseTorcsWrapper] reset failed, forcing relaunch: {exc}")
                if getattr(self.env, "client", None) is not None:
                    self.env.client = None
                relaunch = True

        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        act = np.copy(action)

        steer_damping = max(0.40, 1.0 - (self.current_speed_x / 200.0))
        act[0] = act[0] * steer_damping
        act[2] = act[2] * 1.0 # optional, switched from 0.1

        if self.env.time_step < self.launch_control_steps:
            act[1] = max(float(act[1]), self.launch_min_throttle)
            act[2] = 0.0

        next_state, reward, done, info = self.env.step(act)

        if self.env.client is not None and getattr(self.env.client, "S", None) is not None:
            try:
                obs = self.env.client.S.d
                speed_x = float(obs.get("speedX", 0.0))
                speed_y = float(obs.get("speedY", 0.0))
                angle = float(obs.get("angle", 0.0))
                track_pos = float(obs.get("trackPos", 0.0))
                dist_raced = float(obs.get("distRaced", 0.0))
                cur_lap_time = float(obs.get("curLapTime", 0.0))
                last_lap_time = float(obs.get("lastLapTime", 0.0))

                self.current_speed_x = max(0.0, speed_x)
                self.ep_speed_sum += speed_x
                self.ep_steps += 1
                self.ep_max_speed = max(self.ep_max_speed, speed_x)

                info["dist_raced"] = dist_raced
                info["cur_lap_time"] = cur_lap_time
                info["last_lap_time"] = last_lap_time
                info["episode_count"] = self.episode_count
                info["speed_x"] = speed_x

                if last_lap_time > 0.0 and not done:
                    done = True
                    info["lap_completed"] = True

                progress = speed_x * np.cos(angle)
                custom_reward = progress
                
                custom_reward -= abs(speed_y) * 1.5 # switched from 3.0
                custom_reward -= abs(speed_x * np.sin(angle)) * 1.0
                
                throttle_mag = act[1]
                
                track_sensors = obs.get("track")
                if track_sensors and len(track_sensors) >= 19:
                    forward_dist = float(track_sensors[9])
                    info["forward_dist"] = forward_dist
                    
                    if forward_dist < 55.0:
                        self.heavy_turn_speed_sum += speed_x
                        self.heavy_turn_steps += 1
                    elif forward_dist < 85.0:
                        self.medium_turn_speed_sum += speed_x
                        self.medium_turn_steps += 1
                    elif forward_dist < STRAIGHT_MIN_DIST:
                        self.light_turn_speed_sum += speed_x
                        self.light_turn_steps += 1

                steer_diff = abs(act[0] - self.prev_steer)
                custom_reward -= (steer_diff * max(speed_x, 10.0)) * 0.1

                if self.env.time_step < self.startup_steps:
                    accel_gain = max(0.0, speed_x - self.prev_speed_x)
                    custom_reward += accel_gain * 2.0  
                    custom_reward -= abs(track_pos) * 1.0  
                    custom_reward -= abs(angle) * 3.0   

                if self.env.time_step < self.launch_control_steps:
                    speed_deficit = max(0.0, self.launch_min_speed - speed_x)
                    custom_reward -= speed_deficit * 2.0  
                    custom_reward += max(0.0, speed_x - self.prev_speed_x) * 2.0  
                    custom_reward += throttle_mag * 15.0  

                self.prev_steer = act[0]
                self.prev_speed_x = speed_x
                self.prev_forward_dist = float(info.get("forward_dist", self.prev_forward_dist))

                custom_reward /= 100.0

                if info.get("damage_delta", 0) > 0:
                    custom_reward -= 10.0
                if info.get("off_track", False):
                    custom_reward -= 25.0

                reward = custom_reward

            except Exception:
                pass

        if done:
            info["ep_mean_speed"] = (self.ep_speed_sum / max(1, self.ep_steps))
            info["ep_max_speed"] = self.ep_max_speed
            info["ep_heavy_turn_mean_speed"] = (
                self.heavy_turn_speed_sum / max(1, self.heavy_turn_steps)
                if self.heavy_turn_steps > 0
                else 0.0
            )
            info["ep_medium_turn_mean_speed"] = (
                self.medium_turn_speed_sum / max(1, self.medium_turn_steps)
                if self.medium_turn_steps > 0
                else 0.0
            )
            info["ep_light_turn_mean_speed"] = (
                self.light_turn_speed_sum / max(1, self.light_turn_steps)
                if self.light_turn_steps > 0
                else 0.0
            )
            info["ep_heavy_turn_steps"] = self.heavy_turn_steps
            info["ep_medium_turn_steps"] = self.medium_turn_steps
            info["ep_light_turn_steps"] = self.light_turn_steps

        if done:
            lap_time = info.get("last_lap_time", 0.0)
            dist_raced = info.get("dist_raced", 0.0)
            cur_lap_time = info.get("cur_lap_time", 0.0)
            
            if lap_time > 0 or dist_raced >= 3610.0:
                final_time = lap_time if lap_time > 0 else cur_lap_time
                print(
                    f"[StartFineTune] LAP COMPLETED: {final_time:.2f}s "
                    f"dist={dist_raced:.0f}m"
                )
            else:
                print(
                    "[StartFineTune] Episode ended: "
                    f"off_track={info.get('off_track')} "
                    f"backward={info.get('backward')} "
                    f"stalled={info.get('stalled')} "
                    f"dist={info.get('dist_raced', 0.0):.0f}m "
                    f"time={info.get('cur_lap_time', 0.0):.1f}s"
                )

        return np.array(next_state, dtype=np.float32), float(reward), bool(done), False, info

    def close(self):
        self.env.close(stop_torcs=True)


class TargetLapTimeCallback(CheckpointCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str,
        target_lap_time: float,
    ):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)
        self.target_lap_time = target_lap_time
        self.best_dist = 0.0
        self.best_lap_time = float("inf")
        self.target_beaten = False
        self.csv_path = Path(save_path) / "start_phase_episode_log.csv"
        self._csv_initialized = self.csv_path.exists()
        self._bootstrap_best_metrics_from_csv()

    def _bootstrap_best_metrics_from_csv(self) -> None:
        if not self.csv_path.exists():
            return
        try:
            with self.csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        lap_time = float(row.get("lap_time", "") or 0.0)
                        dist_raced = float(row.get("dist_raced", "") or 0.0)
                    except ValueError:
                        continue

                    if lap_time > 0 and lap_time < self.best_lap_time:
                        self.best_lap_time = lap_time
                    if dist_raced > self.best_dist:
                        self.best_dist = dist_raced
        except Exception as exc:
            print(f"[Start Callback] warning: could not bootstrap metrics from CSV: {exc}")

    def _write_csv_row(self, row: dict) -> None:
        fieldnames = [
            "episode_count", "dist_raced", "lap_time", "cur_lap_time",
            "off_track", "backward", "stalled", "timeout",
            "ep_mean_speed", "ep_max_speed", "ep_heavy_turn_mean_speed",
            "ep_medium_turn_mean_speed", "ep_light_turn_mean_speed",
            "ep_heavy_turn_steps", "ep_medium_turn_steps", "ep_light_turn_steps",
        ]
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    def _on_step(self) -> bool:
        super()._on_step()

        dones = self.locals.get("dones", [False])
        done_flag = bool(dones[0]) if isinstance(dones, (list, np.ndarray)) else bool(dones)
        if not done_flag:
            return True

        info = self.locals.get("infos", [{}])[0]
        dist_raced = float(info.get("dist_raced", 0.0))
        lap_time = float(info.get("last_lap_time", 0.0))
        cur_lap_time = float(info.get("cur_lap_time", 0.0))
        episode_count = int(info.get("episode_count", 0))

        self._write_csv_row({
            "episode_count": episode_count, "dist_raced": dist_raced,
            "lap_time": lap_time, "cur_lap_time": cur_lap_time,
            "off_track": bool(info.get("off_track", False)),
            "backward": bool(info.get("backward", False)),
            "stalled": bool(info.get("stalled", False)),
            "timeout": bool(info.get("timeout", False)),
            "ep_mean_speed": float(info.get("ep_mean_speed", 0.0)),
            "ep_max_speed": float(info.get("ep_max_speed", 0.0)),
            "ep_heavy_turn_mean_speed": float(info.get("ep_heavy_turn_mean_speed", 0.0)),
            "ep_medium_turn_mean_speed": float(info.get("ep_medium_turn_mean_speed", 0.0)),
            "ep_light_turn_mean_speed": float(info.get("ep_light_turn_mean_speed", 0.0)),
            "ep_heavy_turn_steps": int(info.get("ep_heavy_turn_steps", 0)),
            "ep_medium_turn_steps": int(info.get("ep_medium_turn_steps", 0)),
            "ep_light_turn_steps": int(info.get("ep_light_turn_steps", 0)),
        })

        if dist_raced > self.best_dist:
            self.best_dist = dist_raced
            self.model.save(str(Path(self.save_path) / "sac_best_distance_fallback"))

        if dist_raced >= 3608.45 or lap_time > 0.0:
            
            actual_lap_time = lap_time if lap_time > 0.0 else cur_lap_time
            
            if 0.0 < actual_lap_time < self.best_lap_time:
                self.best_lap_time = actual_lap_time
                
                absolute_best_path = Path(self.save_path) / "sac_absolute_best_lap"
                self.model.save(str(absolute_best_path))
                
                print(f"\nNEW ABSOLUTE BEST LAP RECORDED: {self.best_lap_time:.2f}s (dist: {dist_raced:.1f}m) -> Saved to sac_absolute_best_lap.zip")

                snapshot_path = Path(self.save_path) / f"sac_snapshot_{self.best_lap_time:.2f}s"
                self.model.save(str(snapshot_path))

            if self.best_lap_time < self.target_lap_time:
                self.target_beaten = True
                print(f"\nTARGET TIME BEATEN: {self.best_lap_time:.2f}s! Halting training.")
                return False

        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Full-track fine-tuning for SAC TORCS")

    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/checkpoints_sac/exp2/sac_finetune_best_lap.zip",
    )
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default="checkpoints/checkpoints_sac/exp2"
    )
    parser.add_argument(
        "--target-lap-time",
        type=float,
        default=100.0,
        help="Training stops only when a completed lap beats this time (seconds).",
    )
    parser.add_argument(
        "--timesteps-per-iter",
        type=int,
        default=120_000,
        help="Training timesteps per learn() call while searching for target lap time.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=0,
        help="Safety cap on iterations. 0 means unlimited until target is beaten.",
    )

    parser.add_argument(
        "--port", 
        type=int, 
        default=3001
    )
    parser.add_argument(
        "--vision", 
        action="store_true"
    )
    parser.add_argument(
        "--relaunch-every", 
        type=int, 
        default=5
    )
    parser.add_argument(
        "--torcs-command", 
        type=str, 
        default="wine wtorcs.exe"
    )
    parser.add_argument(
        "--torcs-dir", 
        type=str, 
        default="torcs"
    )
    parser.add_argument(
        "--autostart-script", 
        type=str, 
        default="gym_torcs/autostart.sh"
    )
    parser.add_argument(
        "--startup-steps", 
        type=int, 
        default=220
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100_000,
        help="Large per-episode step cap to allow full-lap completion.",
    )

    return parser.parse_args()


def make_env(args):
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
    return StartPhaseTorcsWrapper(
        base_env,
        startup_steps=args.startup_steps,
    )


def set_conservative_hparams(model: SAC):
    model.learning_rate = 2e-5
    for opt in (model.actor.optimizer, model.critic.optimizer):
        for pg in opt.param_groups:
            pg["lr"] = 2e-5
    if hasattr(model, "lr_schedule"):
        model.lr_schedule = lambda _: 2e-5
    model.gamma = 0.995
    model.tau = 0.005


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Creating startup fine-tune environment...")
    env = make_env(args)

    print(f"Loading SAC checkpoint: {args.model_path}")
    model = SAC.load(
        args.model_path,
        env=env,
        tensorboard_log=str(ckpt_dir / "logs_start"),
    )

    set_conservative_hparams(model)

    callback = TargetLapTimeCallback(
        save_freq=5000,
        save_path=str(ckpt_dir),
        name_prefix="sac_start_ckpt",
        target_lap_time=args.target_lap_time,
    )

    iteration = 0
    try:
        while not callback.target_beaten:
            iteration += 1
            print(
                f"\n===== Start Phase Iteration {iteration} | "
                f"timesteps={args.timesteps_per_iter:,} | target<{args.target_lap_time:.2f}s ====="
            )

            model.set_env(env)
            model.learn(
                total_timesteps=args.timesteps_per_iter,
                callback=callback,
                reset_num_timesteps=False,
            )

            iter_path = ckpt_dir / f"sac_start_iter_{iteration:03d}"
            model.save(str(iter_path))
            print(f"Saved iteration checkpoint: {iter_path}.zip")

            if args.max_iters > 0 and iteration >= args.max_iters and not callback.target_beaten:
                print(
                    f"Reached max iterations ({args.max_iters}) without beating "
                    f"{args.target_lap_time:.2f}s."
                )
                break

        if callback.target_beaten:
            print(
                f"\nTarget achieved. Best lap time: {callback.best_lap_time:.2f}s "
                f"(goal: < {args.target_lap_time:.2f}s)."
            )
        else:
            print(
                f"\nTraining ended without target. Current best lap time: "
                f"{callback.best_lap_time:.2f}s"
            )

    except KeyboardInterrupt:
        interrupted_path = ckpt_dir / "sac_start_interrupted"
        model.save(str(interrupted_path))
        print(f"\nStartup fine-tuning interrupted. Saved: {interrupted_path}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()