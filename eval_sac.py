import argparse
from stable_baselines3 import SAC
from src.torcs_env import TorcsRLEnv
from src.torcs_process import TorcsProcessConfig, TorcsProcessManager
from train_sac import GymTorcsWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained TORCS AI using SAC")
    parser.add_argument("--model-path", type=str, default="checkpoints/checkpoints_sac/sac_interrupted.zip",
                        help="Path to the trained model zip file.")
    parser.add_argument("--port", type=int, default=3001)
    
    parser.add_argument("--vision", action="store_true")
    
    parser.add_argument("--torcs-command", type=str, default="wine wtorcs.exe",
                        help="Launch command, e.g. 'torcs' or 'wine wtorcs.exe'.")
    parser.add_argument("--torcs-dir", type=str, default="torcs",
                        help="Working directory for launch command.")
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
    
    print("Initializing environment...")
    base_env = TorcsRLEnv(
        process_manager=process_manager,
        port=args.port,
        vision=args.vision,
        relaunch_every=100,
        max_steps=100000,
    )
    
    env = GymTorcsWrapper(base_env)

    print(f"Loading Model from {args.model_path}...")
    try:
        model = SAC.load(args.model_path, env=env)
    except Exception as e:
        print(f"Failed to load model from {args.model_path}. Error: {e}")
        env.close()
        return

    print("Starting Evaluation Loop! Watch the TORCS window.")
    
    try:
        step_count = 0
        total_reward = 0
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if step_count % 50 == 0:
                print(f"Step: {step_count} | Speed: {info.get('speedX', 0):.1f} | Dist: {info.get('dist_raced', 0):.1f}m | Action: Steer {action[0]:.2f}, Gas {action[1]:.2f}, Brake {action[2]:.2f}")
                
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        print(f"Evaluation finished. Total steps: {step_count}, Total reward: {total_reward}")
        env.close()

if __name__ == "__main__":
    main()
