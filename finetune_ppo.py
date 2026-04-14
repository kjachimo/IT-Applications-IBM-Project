import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from src.torcs_env import TorcsRLEnv
from src.torcs_process import TorcsProcessConfig, TorcsProcessManager


class GymTorcsWrapper(gym.Env):
    """
    Wraps the custom TorcsRLEnv from 'src.torcs_env' into a standard gym.Env
    so it can be consumed natively by Stable-Baselines3 algorithms like PPO.
    """
    def __init__(self, torcs_env: TorcsRLEnv):
        super(GymTorcsWrapper, self).__init__()
        self.env = torcs_env
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.env.state_dim,), 
            dtype=np.float32
        )

        self.episode_count = 0
        self.relaunch_every = self.env.relaunch_every
        self.best_lap_time = float('inf')
        self.prev_steer = 0.0
        self.current_speed_x = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        relaunch = self.episode_count == 1 or (self.episode_count - 1) % self.relaunch_every == 0
        
        while True:
            try:
                state = self.env.reset(relaunch=relaunch)
                self.prev_steer = 0.0
                self.current_speed_x = 0.0
                break
            except Exception as e:
                print(f"[GymTorcsWrapper] Base env reset failed. Forcing relaunch... Error: {e}")
                if getattr(self.env, 'client', None) is not None:
                    self.env.client = None
                relaunch = True 
        
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        act = np.copy(action)
        
        # Speed-sensitive steering: structurally prevent PPO exploration noise from causing spin-outs at high speeds
        steer_damping = max(0.15, 1.0 - (self.current_speed_x / 120.0))
        act[0] = act[0] * steer_damping
        
        act[2] = act[2] * 0.1 
        if self.env.time_step < 10:
            act[1] = 1.0  
            
        next_state, reward, done, info = self.env.step(act)
        
        if self.env.client is not None and getattr(self.env.client, 'S', None) is not None:
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
                
                info['cur_lap_time'] = cur_lap_time
                info['last_lap_time'] = last_lap_time
                info['dist_raced'] = dist_raced
                info['episode_count'] = self.episode_count
                
                if last_lap_time > 0 and last_lap_time < self.best_lap_time:
                    self.best_lap_time = last_lap_time
                    info['best_lap_time'] = self.best_lap_time

                progress = speed_x * np.cos(angle)
                custom_reward = progress
                
                # 1. Penalize lateral sliding (losing traction) heavily
                custom_reward -= abs(speed_y) * 3.0 
                custom_reward -= abs(speed_x * np.sin(angle)) * 2.0
                
                # 1.5. Penalize pointing the car away from the track axis
                custom_reward -= abs(angle) * speed_x * 0.4
                
                # 2. Penalize going off-center at speed
                custom_reward -= (speed_x * (abs(track_pos) ** 3) * 2.5)
                
                # 3. Penalize mashing the throttle while steering
                steering_mag = abs(act[0])
                throttle_mag = act[1]
                if throttle_mag > 0.3 and steering_mag > 0.2:
                    custom_reward -= (throttle_mag * steering_mag * speed_x) * 0.5
                
                # 4. Smooth speed penalty based on straight track ahead
                track_sensors = obs.get("track")
                if track_sensors and len(track_sensors) >= 19:
                    forward_dist = float(track_sensors[9])
                    safe_speed = max(45.0, forward_dist * 1.2)
                    if speed_x > safe_speed:
                        custom_reward -= (speed_x - safe_speed) * 1.5

                # 5. Penalize steering jitter
                steer_diff = abs(act[0] - self.prev_steer)
                custom_reward -= (steer_diff * max(speed_x, 10.0)) * 0.4
                
                # 6. Penalize steering heavily on straights
                custom_reward -= (steering_mag**2 * max(speed_x, 10.0)) * 0.4
                
                self.prev_steer = act[0]

                custom_reward /= 100.0
                
                if info.get("damage_delta", 0) > 0:
                    custom_reward -= 2.0
                    
                if info.get("off_track", False):
                    custom_reward -= 5.0
                    
                reward = custom_reward
            except Exception:
                pass
                
        if done:
            print(f"[PPO Custom] Episode done: off_track={info.get('off_track')} backward={info.get('backward')} stalled={info.get('stalled')} timeout={info.get('timeout')}")
            if 'last_lap_time' in info and info['last_lap_time'] > 0:
                reward += 100.0   
                print(f"[PPO Custom] LAP COMPLETED: Last Lap Time: {info['last_lap_time']}s")
            elif 'cur_lap_time' in info and info['cur_lap_time'] > 0:
                print(f"[PPO Custom] Time survived: {info['cur_lap_time']}s (Dist: {info.get('dist_raced', 0)}m)")

        return np.array(next_state, dtype=np.float32), float(reward), bool(done), False, info

    def close(self):
        self.env.close(stop_torcs=True)


class TensorboardLapTimeCallback(CheckpointCallback):
    """
    Custom callback to log info variables like Lap Time into TensorBoard seamlessly.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = 'ppo_torcs'):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)
        self.best_dist = 0.0
    
    def _on_step(self) -> bool:
        super()._on_step()
        
        done = self.locals.get("dones", [False])[0]
        if done:
            info = self.locals.get("infos", [{}])[0]
            dist_raced = info.get("dist_raced", 0.0)
            episode_count = info.get("episode_count", 0)
            lap_time = info.get("cur_lap_time", 0.0)
            
            writer = None
            for fmt in self.logger.output_formats:
                if hasattr(fmt, "writer"):
                    writer = fmt.writer
                    break
                    
            if writer is not None and episode_count > 0:
                writer.add_scalar("race_per_run/dist_meters", dist_raced, episode_count)
                writer.add_scalar("race_per_run/lap_time_sec", lap_time, episode_count)

            if dist_raced > self.best_dist:
                self.best_dist = dist_raced
                best_model_path = Path(self.save_path) / "finetuned_ppo_best_distance"
                self.model.save(str(best_model_path))
                print(f"\n[Callback] New best distance: {self.best_dist:.2f}m! Model saved to finetuned_ppo_best_distance.zip")
                if writer is not None:
                    writer.add_scalar("race_per_run/best_dist_overall", self.best_dist, episode_count)

            if "last_lap_time" in info and info["last_lap_time"] > 0:
                print(f"\nSUCCESS! The car completed a full lap in {info['last_lap_time']} seconds! ")
                print(f"Max distance recorded overall: {self.best_dist:.2f}m")
                print("Stopping training early to allow further training from this best checkpoint...\n")
                best_lap_path = Path(self.save_path) / "finetuned_ppo_full_lap_success"
                self.model.save(str(best_lap_path))
                return False
            
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a TORCS AI using PPO (On-Policy)")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--relaunch-every", type=int, default=6)
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--vision", action="store_true")
    
    parser.add_argument("--torcs-command", type=str, default="torcs",
                        help="Launch command, e.g. 'torcs' or 'wine wtorcs.exe'.")
    parser.add_argument("--torcs-dir", type=str, default="",
                        help="Working directory for launch command.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo")
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--autostart-script", type=str, default="gym_torcs/autostart.sh")
    
    # Fine-tuning specific arguments
    parser.add_argument("--model-path", type=str, default="checkpoints_ppo/exp4/ppo_best_distance.zip",
                        help="Path to the checkpoint zip file to load.")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                        help="Lower learning rate for fine-tuning.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="Lower entropy coefficient to reduce exploration noise.")
    parser.add_argument("--clip-range", type=float, default=0.05,
                        help="Tighten PPO clipping to prevent destructive updates to the good policy.")
    parser.add_argument("--n-epochs", type=int, default=3,
                        help="Fewer epochs to prevent overfitting to recent noisy batches.")
    
    return parser.parse_args()


def main():
    args = parse_args()

    print("Configuring Process Manager for Fine-Tuning...")
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
        max_steps=100000,
    )
    
    env = GymTorcsWrapper(base_env)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {args.model_path}. Provide valid --model-path.")

    print(f"Applying Fine-Tuning Hyperparameters: LR={args.learning_rate}, ENT={args.ent_coef}, CLIP={args.clip_range}, EPOCHS={args.n_epochs}")
    
    custom_objects = {
        "learning_rate": args.learning_rate,
        "ent_coef": args.ent_coef,
        "clip_range": args.clip_range,
        "n_epochs": args.n_epochs,
    }
        
    print(f"Loading existing PPO Agent from {model_path}...")
    model = PPO.load(
        str(model_path), 
        env=env, 
        tensorboard_log=str(ckpt_dir / "logs"), 
        custom_objects=custom_objects
    )
    
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = args.learning_rate

    checkpoint_callback = TensorboardLapTimeCallback(
        save_freq=args.checkpoint_every, 
        save_path=str(ckpt_dir), 
        name_prefix='finetuned_ppo_torcs'
    )

    print(f"Starting Fine-Tuning for {args.timesteps} timesteps...")
    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, reset_num_timesteps=False)
        model.save(str(ckpt_dir / "finetuned_ppo_final"))
        print("Training Completed and Model Saved!")
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")
        model.save(str(ckpt_dir / "finetuned_ppo_interrupted"))
    finally:
        env.close()

if __name__ == "__main__":
    main()
