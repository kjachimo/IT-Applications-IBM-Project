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
        
        # State Dim = 29
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.env.state_dim,), 
            dtype=np.float32
        )

        self.episode_count = 0
        self.relaunch_every = self.env.relaunch_every

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        relaunch = self.episode_count == 1 or (self.episode_count - 1) % self.relaunch_every == 0
        
        while True:
            try:
                state = self.env.reset(relaunch=relaunch)
                break
            except Exception as e:
                print(f"[GymTorcsWrapper] Base env reset failed. Forcing relaunch... Error: {e}")
                if getattr(self.env, 'client', None) is not None:
                    self.env.client = None
                relaunch = True 
        
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        act = np.copy(action)
        act[2] = act[2] * 0.1 
        if self.env.time_step < 10:
            act[1] = 1.0  
            
        next_state, reward, done, info = self.env.step(act)
        
        if self.env.client is not None and getattr(self.env.client, 'S', None) is not None:
            try:
                obs = self.env.client.S.d
                speed_x = float(obs.get("speedX", 0))
                angle = float(obs.get("angle", 0))
                track_pos = float(obs.get("trackPos", 0))
                
                progress = speed_x * np.cos(angle)
                custom_reward = progress - abs(speed_x * np.sin(angle))
                
                custom_reward -= (speed_x * (abs(track_pos) ** 3) * 2.5)
                
                track_sensors = obs.get("track")
                if track_sensors and len(track_sensors) >= 19:
                    forward_dist = float(track_sensors[9])
                    if forward_dist < 70.0 and speed_x > 60.0:
                        custom_reward -= (speed_x - 60.0) * 0.5

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

        return np.array(next_state, dtype=np.float32), float(reward), bool(done), False, info

    def close(self):
        self.env.close(stop_torcs=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TORCS AI using PPO (On-Policy)")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--relaunch-every", type=int, default=3)
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--vision", action="store_true")
    
    parser.add_argument("--torcs-command", type=str, default="torcs",
                        help="Launch command, e.g. 'torcs' or 'wine wtorcs.exe'.")
    parser.add_argument("--torcs-dir", type=str, default="",
                        help="Working directory for launch command.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--autostart-script", type=str, default="gym_torcs/autostart.sh")
    
    return parser.parse_args()


def main():
    args = parse_args()

    print("Configuring Process Manager...")
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
        max_steps=2000,
    )
    
    env = GymTorcsWrapper(base_env)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing PPO Agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        ent_coef=0.01,          
        learning_rate=1e-4,
        n_steps=2048,          
        batch_size=64,
        n_epochs=10,         
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log=str(ckpt_dir / "logs") 
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_every, 
        save_path=str(ckpt_dir), 
        name_prefix='ppo_torcs'
    )

    print(f"Starting Training for {args.timesteps} timesteps...")
    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
        model.save(str(ckpt_dir / "ppo_final"))
        print("Training Completed and Model Saved!")
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")
        model.save(str(ckpt_dir / "ppo_interrupted"))
    finally:
        env.close()

if __name__ == "__main__":
    main()