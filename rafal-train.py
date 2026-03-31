import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.ddpg import DDPGAgent
from src.torcs_env import TorcsRLEnv
from src.torcs_process import TorcsProcessConfig, TorcsProcessManager


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TORCS AI driver with DDPG")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--relaunch-every", type=int, default=3)
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--torcs-command",
        type=str,
        default="torcs",
        help="Launch command, e.g. 'torcs' or 'wine wtorcs.exe'.",
    )
    parser.add_argument(
        "--torcs-dir",
        type=str,
        default="",
        help="Working directory for launch command, useful with Wine .exe runs.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--track-name", type=str, default="corkscrew")
    parser.add_argument("--track-category", type=str, default="road")
    parser.add_argument(
        "--autostart-script",
        type=str,
        default="gym_torcs/autostart.sh",
        help="Script that navigates TORCS menus and starts a race.",
    )
    parser.add_argument("--stop-torcs-on-exit", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    print(
        f"Training target: track={args.track_name}, category={args.track_category}, default TORCS car"
    )

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
        raise RuntimeError(
            "Missing requirements: " + ", ".join(missing) + ". "
            "Install dependencies and ensure autostart script path is correct."
        )

    env = TorcsRLEnv(
        process_manager=process_manager,
        port=args.port,
        vision=args.vision,
        relaunch_every=args.relaunch_every,
        max_steps=args.max_steps,
    )
    agent = DDPGAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        agent.load(args.resume)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    try:
        for episode in range(1, args.episodes + 1):
            relaunch = episode == 1 or (episode - 1) % args.relaunch_every == 0
            state = env.reset(relaunch=relaunch)
            agent.reset_noise()

            episode_reward = 0.0
            actor_losses = []
            critic_losses = []

            for step in range(1, args.max_steps + 1):
                if global_step < args.warmup_steps:
                    action = np.array(
                        [
                            np.random.uniform(-1.0, 1.0),
                            np.random.uniform(0.0, 1.0),
                            np.random.uniform(0.0, 0.3),
                        ],
                        dtype=np.float32,
                    )
                else:
                    action = agent.select_action(state, explore=True)

                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                global_step += 1

                for _ in range(args.updates_per_step):
                    losses = agent.train_step()
                    if losses is not None:
                        actor_losses.append(losses["actor_loss"])
                        critic_losses.append(losses["critic_loss"])

                if done:
                    break

            mean_actor_loss = float(np.mean(actor_losses)) if actor_losses else float("nan")
            mean_critic_loss = (
                float(np.mean(critic_losses)) if critic_losses else float("nan")
            )

            print(
                f"Episode {episode:04d} | steps={step:04d} | reward={episode_reward:8.3f} "
                f"| actor_loss={mean_actor_loss:8.5f} | critic_loss={mean_critic_loss:8.5f} "
                f"| progress={info.get('progress', 0.0):8.3f} | done={done}"
            )

            if episode % args.checkpoint_every == 0:
                ckpt_path = ckpt_dir / f"ddpg_ep{episode:04d}.pt"
                agent.save(str(ckpt_path))
                print(f"Checkpoint saved: {ckpt_path}")

        final_ckpt = ckpt_dir / "ddpg_final.pt"
        agent.save(str(final_ckpt))
        print(f"Final checkpoint saved: {final_ckpt}")

    finally:
        env.close(stop_torcs=args.stop_torcs_on_exit)


if __name__ == "__main__":
    main()
