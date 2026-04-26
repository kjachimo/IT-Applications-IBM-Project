import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

from src.torcs_env import TorcsRLEnv
from src.torcs_process import TorcsProcessConfig, TorcsProcessManager


class GymTorcsWrapperSpeedOptimized(gym.Env):
    """
    Wraps the custom TorcsRLEnv with a reward function optimized for SPEED and shortest lap times.
    This is tuned for fine-tuning an already-trained model to go faster.
    """

    def __init__(self, torcs_env: TorcsRLEnv):
        super().__init__()
        self.env = torcs_env

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.state_dim,), dtype=np.float32
        )

        self.episode_count = 0
        self.relaunch_every = self.env.relaunch_every
        self.best_lap_time = float("inf")
        self.prev_steer = 0.0
        self.current_speed_x = 0.0
        self.speed_target = 80.0  # Initial conservative target, will increase as model learns
        self.last_lap_times = []  # Track lap times for curriculum adjustment

        # Track average speed for reward calculation
        self.cumulative_speed = 0.0
        self.step_count = 0
        self.episode_reward = 0.0
        self.stationary_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        relaunch = self.episode_count == 1 or (self.episode_count - 1) % self.relaunch_every == 0

        while True:
            try:
                state = self.env.reset(relaunch=relaunch)
                self.prev_steer = 0.0
                self.current_speed_x = 0.0
                self.cumulative_speed = 0.0
                self.step_count = 0
                self.episode_reward = 0.0
                self.stationary_steps = 0
                break
            except Exception as e:
                print(f"[GymTorcsWrapper] Base env reset failed. Forcing relaunch... Error: {e}")
                if getattr(self.env, "client", None) is not None:
                    self.env.client = None
                relaunch = True

        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        act = np.copy(action)

        # Speed-sensitive steering: prevent heavy spin-outs at high speeds, but allow aggressive steering when appropriate
        steer_damping = max(0.15, 1.0 - (self.current_speed_x / 120.0))
        act[0] = act[0] * steer_damping

        act[2] = act[2] * 0.1
        if self.env.time_step < 10:
            act[1] = 1.0

        next_state, reward, done, info = self.env.step(act)

        if self.env.client is not None and getattr(self.env.client, "S", None) is not None:
            try:
                obs = self.env.client.S.d
                speed_x = float(obs.get("speedX", 0))
                speed_y = float(obs.get("speedY", 0))
                angle = float(obs.get("angle", 0))
                track_pos = float(obs.get("trackPos", 0))

                self.current_speed_x = max(0.0, speed_x)

                cur_lap_time = float(obs.get("curLapTime", 0))
                last_lap_time = float(obs.get("lastLapTime", 0))
                dist_raced = float(obs.get("distRaced", 0))

                info["cur_lap_time"] = cur_lap_time
                info["last_lap_time"] = last_lap_time
                info["dist_raced"] = dist_raced
                info["episode_count"] = self.episode_count

                if last_lap_time > 0 and last_lap_time < self.best_lap_time:
                    self.best_lap_time = last_lap_time
                    info["best_lap_time"] = self.best_lap_time
                    self.last_lap_times.append(last_lap_time)

                    # Progressive speed target: target 2% improvement each time
                    self.speed_target = self.best_lap_time * 0.98
                    print(
                        f"[FINETUNE] New best lap time: {self.best_lap_time:.2f}s, target: {self.speed_target:.2f}s"
                    )

                # Track cumulative speed for average speed calculation
                self.cumulative_speed += max(speed_x, 0)
                self.step_count += 1
                avg_speed = self.cumulative_speed / max(self.step_count, 1)

                # ===== RELIABILITY-FIRST REWARD FUNCTION =====
                # Primary objective: stay on track and progress consistently.
                progress = max(0.0, speed_x * np.cos(angle))
                custom_reward = progress * 0.03

                # Encourage stable driving line for consistent lap completion.
                center_bonus = max(0.0, 1.0 - abs(track_pos))
                custom_reward += center_bonus * 0.5

                # Keep average speed signal, but with lower influence than stability.
                custom_reward += avg_speed * 0.01

                # Secondary: Bonus for completing laps faster
                if last_lap_time > 0:
                    lap_bonus = 15.0 / (
                        last_lap_time / 100.0
                    )  # Inverse relation (faster = bigger bonus)
                    custom_reward += lap_bonus

                # Penalize unstable vehicle dynamics and large heading errors.
                # custom_reward -= abs(speed_y) * 0.8
                # custom_reward -= abs(speed_x * np.sin(angle)) * 0.6
                # custom_reward -= abs(angle) * speed_x * 0.1

                # Penalize running wide, but not so strongly that recovery becomes impossible.
                # custom_reward -= speed_x * (abs(track_pos) ** 2) * 0.25

                # 3. Less aggressive on throttle+steering combo (allow for hard braking into turns)
                steering_mag = abs(act[0])
                # throttle_mag = act[1]
                # if throttle_mag > 0.5 and steering_mag > 0.3:
                #     custom_reward -= (
                #         throttle_mag * steering_mag * speed_x
                #     ) * 0.2  # reduced from 0.5

                # Track-aware speed penalty to avoid frequent spin-outs.
                # track_sensors = obs.get("track")
                # if track_sensors and len(track_sensors) >= 19:
                #     forward_dist = float(track_sensors[9])
                #     safe_speed = max(45.0, forward_dist * 1.2)
                #     if speed_x > safe_speed:
                #         custom_reward -= (speed_x - safe_speed) * 0.5

                # Penalize being stationary for too long (with grace period).
                if speed_x < 3.0:
                    self.stationary_steps += 1
                else:
                    self.stationary_steps = 0

                if self.env.time_step > 30 and self.stationary_steps > 15:
                    stuck_penalty = min(2.0, 0.05 * (self.stationary_steps - 15))
                    custom_reward -= stuck_penalty
                    info["stuck_penalty"] = stuck_penalty
                else:
                    info["stuck_penalty"] = 0.0
                info["stationary_steps"] = self.stationary_steps

                # 5. Penalize steering jitter (reduced)
                steer_diff = abs(act[0] - self.prev_steer)
                custom_reward -= (steer_diff * max(speed_x, 10.0)) * 0.2  # reduced from 0.4

                # 6. Penalize steering on straights (removed - allow aggressive steering for tight turns)
                custom_reward -= (
                    steering_mag**2 * max(speed_x, 10.0)
                ) * 0.4  # REMOVED for speed optimization

                self.prev_steer = act[0]

                # Critical safety constraints (keep these)
                if info.get("damage_delta", 0) > 0:
                    custom_reward -= 1.0

                if info.get("off_track", False):
                    custom_reward -= 2.0

                reward = custom_reward
            except Exception:
                pass

        step_reward = float(reward)
        self.episode_reward += step_reward
        info["episode_reward"] = self.episode_reward
        info["avg_step_reward"] = self.episode_reward / max(self.step_count, 1)

        if done:
            avg_speed = self.cumulative_speed / max(self.step_count, 1)
            print(
                f"[FINETUNE] Episode done: off_track={info.get('off_track')} backward={info.get('backward')} stalled={info.get('stalled')} timeout={info.get('timeout')}"
            )
            print(f"[FINETUNE] Average speed this episode: {avg_speed:.2f} m/s")
            print(
                f"[FINETUNE] Stationary steps: {self.stationary_steps} (last stuck penalty: {info.get('stuck_penalty', 0.0):.3f})"
            )
            print(
                f"[FINETUNE] Episode reward: {self.episode_reward:.4f} (avg step reward: {info.get('avg_step_reward', 0.0):.6f})"
            )
            if "last_lap_time" in info and info["last_lap_time"] > 0:
                print(
                    f"[FINETUNE] LAP COMPLETED: Last Lap Time: {info['last_lap_time']:.2f}s (Best: {self.best_lap_time:.2f}s)"
                )
            elif "cur_lap_time" in info and info["cur_lap_time"] > 0:
                print(
                    f"[FINETUNE] Time survived: {info['cur_lap_time']:.2f}s (Dist: {info.get('dist_raced', 0):.2f}m)"
                )

        return (
            np.array(next_state, dtype=np.float32),
            float(reward),
            bool(done),
            False,
            info,
        )

    def close(self):
        self.env.close(stop_torcs=True)


class LapTimeOptimizationCallback(CheckpointCallback):
    """
    Custom callback that saves models based on best lap time (not just distance).
    Fine-tuning for speed optimization.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "sac_finetune"):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)
        self.best_lap_time = float("inf")
        self.best_dist = 0.0
        self.best_episode_reward = -float("inf")

    def _on_step(self) -> bool:
        super()._on_step()

        done = self.locals.get("dones", [False])
        if done[0] if isinstance(done, (list, np.ndarray)) else done:
            info = self.locals.get("infos", [{}])[0]
            dist_raced = info.get("dist_raced", 0.0)
            episode_count = info.get("episode_count", 0)
            lap_time = info.get("last_lap_time", 0.0)
            episode_reward = info.get("episode_reward", 0.0)
            avg_step_reward = info.get("avg_step_reward", 0.0)

            writer = None
            for fmt in self.logger.output_formats:
                writer = getattr(fmt, "writer", None)
                if writer is not None:
                    break

            if writer is not None and episode_count > 0:
                writer.add_scalar("finetune/dist_meters", dist_raced, episode_count)
                writer.add_scalar("finetune/lap_time_sec", lap_time, episode_count)
                writer.add_scalar("finetune/episode_reward", episode_reward, episode_count)
                writer.add_scalar("finetune/avg_step_reward", avg_step_reward, episode_count)

            if episode_reward > self.best_episode_reward:
                self.best_episode_reward = episode_reward
                print(
                    f"[CALLBACK] New best episode reward: {self.best_episode_reward:.4f} at episode {episode_count}"
                )
                if writer is not None:
                    writer.add_scalar(
                        "finetune/best_episode_reward",
                        self.best_episode_reward,
                        episode_count,
                    )

            # Save model when distance improves
            if dist_raced > self.best_dist:
                self.best_dist = dist_raced
                best_model_path = Path(self.save_path) / "sac_finetune_best_distance"
                self.model.save(str(best_model_path))
                print(
                    f"\n[CALLBACK] New best distance: {self.best_dist:.2f}m! Model saved to sac_finetune_best_distance.zip"
                )
                if writer is not None:
                    writer.add_scalar("finetune/best_dist_overall", self.best_dist, episode_count)

            # Save model when lap time improves (primary objective for fine-tuning)
            if lap_time > 0 and lap_time < self.best_lap_time:
                self.best_lap_time = lap_time
                best_lap_path = Path(self.save_path) / "sac_finetune_best_laptime"
                self.model.save(str(best_lap_path))
                print(
                    f"\n[CALLBACK] New best lap time: {self.best_lap_time:.2f}s! Model saved to sac_finetune_best_laptime.zip"
                )
                if writer is not None:
                    writer.add_scalar("finetune/best_lap_time", self.best_lap_time, episode_count)

        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a TORCS SAC agent for SPEED (shortest lap times). "
        "Start from an already-trained model that can complete laps."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Number of fine-tuning steps (default: 500k)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the pre-trained SAC model checkpoint to fine-tune (e.g., checkpoints_sac/sac_full_lap_success)",
    )
    parser.add_argument("--relaunch-every", type=int, default=5)
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--vision", action="store_true")

    parser.add_argument(
        "--torcs-command",
        type=str,
        default="wine wtorcs.exe",
        help="Launch command, e.g. 'torcs' or 'wine wtorcs.exe'.",
    )
    parser.add_argument(
        "--torcs-dir",
        type=str,
        default="torcs",
        help="Working directory for launch command.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_sac_finetune")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="How often to run evaluation (in training steps).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=2,
        help="How many episodes per evaluation run.",
    )
    parser.add_argument(
        "--eval-patience",
        type=int,
        default=5,
        help="Stop after this many eval runs without improvement.",
    )
    parser.add_argument("--autostart-script", type=str, default="gym_torcs/autostart.sh")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("FINE-TUNING SAC AGENT FOR SPEED (SHORTEST LAP TIMES)")
    print("=" * 80)
    print(f"Loading pre-trained model from: {args.model}")
    print(f"Fine-tuning for {args.timesteps} timesteps")
    print(f"Results will be saved to: {args.checkpoint_dir}")
    print("=" * 80)

    print("\nConfiguring Process Manager...")
    process_manager = TorcsProcessManager(
        autostart_script=args.autostart_script,
        config=TorcsProcessConfig(
            torcs_command=args.torcs_command,
            torcs_working_dir=args.torcs_dir,
            vision=args.vision,
        ),
    )

    missing = process_manager.check_requirements()
    if missing:
        raise RuntimeError("Missing requirements: " + ", ".join(missing))

    print("Initializing environment...")
    base_env = TorcsRLEnv(
        process_manager=process_manager,
        port=args.port,
        vision=args.vision,
        relaunch_every=args.relaunch_every,
        max_steps=args.timesteps,  # shorter episodes for fine-tuning
    )

    env = GymTorcsWrapperSpeedOptimized(base_env)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-trained model
    print(f"\nLoading pre-trained SAC model from {args.model}...")
    model = SAC.load(args.model, env=env, tensorboard_log=str(ckpt_dir / "logs"))

    print("\nFine-tuning model for SPEED optimization (EXPLOITATION-focused)...")
    print("Using loaded model configuration with:")
    print("  - Reliability-first reward shaping")
    print("  - Periodic evaluation runs on the same env")
    print("  - Early stop if eval does not improve")
    print(f"  - Entropy Coefficient: {model.ent_coef}")

    print(f"  - Current Learning Rate: {model.learning_rate}")
    print(f"  - Gamma (discount): {model.gamma}")
    print(f"  - Tau (soft update): {model.tau}")

    checkpoint_callback = LapTimeOptimizationCallback(
        save_freq=args.checkpoint_every,
        save_path=str(ckpt_dir),
        name_prefix="sac_finetune",
    )

    stop_no_improvement_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=args.eval_patience,
        min_evals=3,
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env=env,
        callback_after_eval=stop_no_improvement_callback,
        best_model_save_path=str(ckpt_dir / "eval_best"),
        log_path=str(ckpt_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    print(f"\nStarting Fine-tuning for {args.timesteps} timesteps...")
    print("Objective: Minimize lap time by optimizing for speed\n")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=False,  # Continue from original training step count
        )
        model.save(str(ckpt_dir / "sac_finetune_final"))
        print("\nFine-tuning Completed and Final Model Saved!")
        print(f"Best models saved in: {ckpt_dir}")
        print("  - sac_finetune_best_laptime.zip (fastest lap time)")
        print("  - sac_finetune_best_distance.zip (longest distance)")
        print("  - sac_finetune_final.zip (final state)")
    except KeyboardInterrupt:
        print("\nFine-tuning interrupted! Saving current model...")
        model.save(str(ckpt_dir / "sac_finetune_interrupted"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
