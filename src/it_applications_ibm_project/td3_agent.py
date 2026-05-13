from typing import Any

from it_applications_ibm_project.driver_action import ActionData
from it_applications_ibm_project.server_state import SensorData
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        steer = self.max_action * torch.tanh(a[:, 0:1])
        accel = torch.sigmoid(a[:, 1:2])
        brake = torch.sigmoid(a[:, 2:3])
        return torch.cat([steer, accel, brake], dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)


import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state_to_tensor(state).numpy()[0]
        self.action[self.ptr] = action_to_tensor(action).numpy()[0]
        self.next_state[self.ptr] = state_to_tensor(next_state).numpy()[0]
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def clear(self):
        self.ptr = 0
        self.size = 0

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[idx]),
            torch.FloatTensor(self.action[idx]),
            torch.FloatTensor(self.next_state[idx]),
            torch.FloatTensor(self.reward[idx]),
            torch.FloatTensor(self.done[idx]),
        )


class TD3(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        super().__init__()
        print(
            f"Initializing TD3 with state_dim={state_dim}, action_dim={action_dim}, max_action={max_action}"
        )
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state: SensorData):
        state_t = state_to_tensor(state)
        action_t = self.actor(state_t).detach().numpy()[0]
        return tensor_to_action(torch.FloatTensor(action_t.reshape(1, -1)))

    def train(self, replay_buffer, batch_size=256):
        print("Training TD3...")
        self.total_it += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        print(f"Loading TD3 model from {path}...")
        self.load_state_dict(torch.load(path))


def state_to_tensor(state: SensorData) -> torch.FloatTensor:
    state_t = torch.tensor(
        [
            state["angle"],
            state["distFromStart"],
            state["distRaced"],
            state["speedX"],
            state["speedY"],
            state["speedZ"],
            *state["track"],
            state["trackPos"],
            *state["wheelSpinVel"],
            state["z"],
        ],
        dtype=torch.float32,
    )
    return torch.FloatTensor(state_t.reshape(1, -1))


def tensor_to_action(action_tensor) -> ActionData:
    action_array = action_tensor.detach().numpy()[0]
    return {
        "steer": action_array[0],
        "accel": action_array[1],
        "brake": action_array[2],
    }


def action_to_tensor(action: ActionData) -> torch.Tensor:
    return torch.tensor(
        [action.get("steer", 0.0), action.get("accel", 0.0), action.get("brake", 0.0)],
        dtype=torch.float32,
    )
