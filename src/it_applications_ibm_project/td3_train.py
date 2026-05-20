from it_applications_ibm_project.gym_torcs import TorcsEnv
from it_applications_ibm_project.td3_agent import (
    TD3,
    ReplayBuffer,
    state_to_tensor,
    action_to_tensor,
)
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from it_applications_ibm_project.torcs_utils import automatic_transmission
from it_applications_ibm_project.driver_action import DriverAction
from it_applications_ibm_project.expert import drive_modular
import argparse
import numpy as np
from pathlib import Path

SAVE_DIR = Path("models/td3")
LAST_MODEL_PATH = SAVE_DIR / "td3_actor_last.pth"
BEST_MODEL_PATH = SAVE_DIR / "td3_actor_best.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TD3 in TORCS")
    parser.add_argument(
        "-p",
        "--pretrained",
        dest="pretrained_path",
        default=None,
        help="Path to a pretrained TD3 model to load before training",
    )
    parser.add_argument(
        "--bc-pretrain",
        action="store_true",
        help="Run behavioral cloning pretraining from the deterministic expert before RL",
    )
    parser.add_argument(
        "--bc-samples",
        type=int,
        default=2000,
        help="Number of expert samples to collect for BC pretraining",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=10,
        help="Number of epochs for BC pretraining",
    )
    parser.add_argument(
        "--bc-batch",
        type=int,
        default=64,
        help="Batch size for BC pretraining",
    )
    args = parser.parse_args()

    vision = False
    episode_count = 100
    max_steps = 5000
    best_reward = -float("inf")
    reward = 0
    done = False
    step = 0

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    env = TorcsEnv(vision=vision)

    model = TD3(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=1,
    )
    if args.pretrained_path:
        model.load(args.pretrained_path)
    replay_buffer = ReplayBuffer(
        state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]
    )

    def collect_expert_data(env: TorcsEnv, num_samples: int):
        states = []
        actions = []
        ob = env.reset()
        while len(states) < num_samples:
            # Start with a default action dict
            expert_action = {"steer": 0.0, "accel": 0.0, "brake": 0.0}
            expert_action = drive_modular(ob, expert_action)
            expert_action["gear"] = automatic_transmission(ob)
            da = DriverAction()
            da.d = expert_action
            da.clip_to_limits()
            expert_action = da.d

            next_ob, reward, done, _ = env.step(expert_action)

            states.append(state_to_tensor(ob))
            actions.append(action_to_tensor(expert_action))

            ob = next_ob
            if done:
                ob = env.reset()

        states_t = torch.cat(states, dim=0)
        actions_t = torch.cat(actions, dim=0)
        return states_t, actions_t

    def bc_pretrain(
        model: TD3,
        states_t: torch.Tensor,
        actions_t: torch.Tensor,
        epochs: int,
        batch_size: int,
    ):
        dataset = TensorDataset(states_t, actions_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        mse = nn.MSELoss()
        actor = model.actor
        optim = model.actor_optimizer
        actor.train()
        for e in range(epochs):
            epoch_loss = 0.0
            for s_batch, a_batch in loader:
                pred = actor(s_batch)
                loss = mse(pred, a_batch)
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item() * s_batch.size(0)
            epoch_loss = epoch_loss / len(dataset)
            print(f"BC Epoch {e + 1}/{epochs} loss: {epoch_loss:.6f}")

    print("TORCS Experiment Start.")

    # Behavioral cloning pretraining (optional)
    if args.bc_pretrain:
        print("Collecting expert data for BC pretraining...")
        states_t, actions_t = collect_expert_data(env, args.bc_samples)
        print(
            f"Collected {states_t.shape[0]} expert samples. Starting BC pretraining..."
        )
        bc_pretrain(model, states_t, actions_t, args.bc_epochs, args.bc_batch)
        model.save(LAST_MODEL_PATH.as_posix())
    for i in range(episode_count):
        print("Episode : " + str(i))

        if np.mod(i, 5) == 0:
            ob = env.reset(relaunch=True)
            model.reset_noise()
        else:
            ob = env.reset()
            model.reset_noise()

        total_reward = 0.0
        for j in range(max_steps):
            action = model.select_action_noisy(ob, noise=True)
            if j == 0:
                expert_action = action.copy()
            action["gear"] = automatic_transmission(ob)
            driver_action = DriverAction()
            driver_action.d = action
            driver_action.clip_to_limits()
            action = driver_action.d

            next_ob, reward, done, _ = env.step(action)

            total_reward += reward

            expert_action = drive_modular(ob, expert_action)

            replay_buffer.add(ob, action, next_ob, reward, done)
            # print(
            #     f"Step {j} - Action: {action}, Reward: {reward:.3f}, Total Reward: {total_reward:.3f}"
            # )

            ob = next_ob

            if replay_buffer.size > 1000:
                model.train(replay_buffer, batch_size=256)
                replay_buffer.clear()

            step += 1
            if done:
                model.train(replay_buffer, batch_size=256)
                break

        print("TOTAL REWARD @ " + str(i) + " -th Episode  :  " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

        if total_reward > best_reward:
            best_reward = total_reward
            model.save(BEST_MODEL_PATH.as_posix())

        model.save(LAST_MODEL_PATH.as_posix())

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    main()
