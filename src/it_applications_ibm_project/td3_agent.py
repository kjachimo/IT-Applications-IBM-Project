from typing import Any

from it_applications_ibm_project.driver_action import ActionData
from it_applications_ibm_project.server_state import SensorData
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        ## Inspired by https://github.com/yanpanlau/DDPG-Keras-Torcs/blob/master/ActorNetwork.py
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 512)
        self.l_S = nn.Linear(512, 1)  # Steering output
        self.l_A = nn.Linear(512, 1)  # Acceleration output
        self.l_B = nn.Linear(512, 1)  # Brake output
        #
        nn.init.normal_(self.l_S.weight, -1e-4, 1e-4)
        nn.init.normal_(self.l_A.weight, -1e-4, 1e-4)
        nn.init.normal_(self.l_B.weight, -1e-4, 1e-4)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        steer = torch.tanh(self.l_S(a))  # Steering in range [-1, 1]
        accel = torch.sigmoid(self.l_A(a))  # Acceleration in range [0, 1]
        brake = torch.sigmoid(self.l_B(a))  # Brake in range [0, 1]
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
        ou_theta=None,
        ou_mu=None,
        ou_sigma=None,
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
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        # Per-dimension action bounds: [steer, accel, brake]
        self.action_low = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32)
        self.action_high = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        # Ornstein-Uhlenbeck process for exploration (per-dimension)
        default_theta = np.array([0.6, 1.0, 1.0], dtype=np.float32)
        default_mu = np.array([0.0, 0.45, -0.1], dtype=np.float32)
        default_sigma = np.array([0.30, 0.10, 0.05], dtype=np.float32)
        self.ou_theta = np.array(ou_theta, dtype=np.float32) if ou_theta is not None else default_theta
        self.ou_mu = np.array(ou_mu, dtype=np.float32) if ou_mu is not None else default_mu
        self.ou_sigma = np.array(ou_sigma, dtype=np.float32) if ou_sigma is not None else default_sigma
        self._ou_state = self.ou_mu.copy()
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state: SensorData):
        return self.select_action_noisy(state, noise=False)

    def select_action_noisy(self, state: SensorData, noise: bool = False):
        state_t = state_to_tensor(state)
        action_t = self.actor(state_t).detach().cpu().numpy()[0]
        if noise:
            # Sample OU noise and add to action
            noise_sample = self._ou_sample()
            action_t = action_t + noise_sample
        # Clip per-dimension to action bounds
        action_t = np.minimum(np.maximum(action_t, self.action_low.numpy()), self.action_high.numpy())
        return tensor_to_action(torch.FloatTensor(action_t.reshape(1, -1)))

    def reset_noise(self):
        # Reset OU internal state to mean
        self._ou_state = self.ou_mu.copy()

    def _ou_sample(self):
        # Discrete-time OU: x_{t+1} = x_t + theta*(mu - x_t) + sigma * N(0,1)
        dx = self.ou_theta * (self.ou_mu - self._ou_state) + self.ou_sigma * np.random.randn(*self._ou_state.shape)
        self._ou_state = self._ou_state + dx
        return self._ou_state

    def train(self, replay_buffer, batch_size=256):
        print("Training TD3...")
        self.total_it += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            # Add noise to the target action and clamp per-dimension so
            # accel/brake remain within [0,1] while steer stays within [-1,1].
            next_action = self.actor_target(next_state) + noise
            action_low = self.action_low.to(next_action.device)
            action_high = self.action_high.to(next_action.device)
            next_action = torch.max(torch.min(next_action, action_high), action_low)

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
    # Return as a 2D tensor with shape (1, action_dim) to match state tensors
    return torch.tensor(
        [[action.get("steer", 0.0), action.get("accel", 0.0), action.get("brake", 0.0)]],
        dtype=torch.float32,
    )
