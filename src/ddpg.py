from collections import deque
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    def __init__(self, theta, mu, sigma):
        self.theta = np.array(theta, dtype=np.float32)
        self.mu = np.array(mu, dtype=np.float32)
        self.sigma = np.array(sigma, dtype=np.float32)
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            *self.mu.shape
        )
        self.state = self.state + dx
        return self.state


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 600),
            nn.ReLU(),
        )
        self.steer = nn.Linear(600, 1)
        self.accel = nn.Linear(600, 1)
        self.brake = nn.Linear(600, 1)

        self._init_output(self.steer)
        self._init_output(self.accel)
        self._init_output(self.brake)

    @staticmethod
    def _init_output(layer):
        nn.init.uniform_(layer.weight, -1e-4, 1e-4)
        nn.init.uniform_(layer.bias, -1e-4, 1e-4)

    def forward(self, state):
        h = self.net(state)
        steer = torch.tanh(self.steer(h))
        accel = torch.sigmoid(self.accel(h))
        brake = torch.sigmoid(self.brake(h))
        return torch.cat([steer, accel, brake], dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_fc = nn.Linear(state_dim, 300)
        self.state_to_hidden = nn.Linear(300, 600)
        self.action_to_hidden = nn.Linear(action_dim, 600)
        self.q_out = nn.Linear(600, 1)

    def forward(self, state, action):
        h1 = torch.relu(self.state_fc(state))
        h_state = self.state_to_hidden(h1)
        h_action = self.action_to_hidden(action)
        h = torch.relu(h_state + h_action)
        return self.q_out(h)


class DDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 1e-3,
        replay_size: int = 100000,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = ReplayBuffer(replay_size)
        self.noise = OUNoise(
            theta=[0.60, 1.00, 1.00],
            mu=[0.0, 0.5, -0.1],
            sigma=[0.30, 0.10, 0.05],
        )

    def reset_noise(self):
        self.noise.reset()

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        if explore:
            action = action + self.noise.sample()

        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip(action[1], 0.0, 1.0)
        action[2] = np.clip(action[2], 0.0, 1.0)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[Dict[str, float]]:
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).float().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.float32)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            y = rewards_t + self.gamma * (1.0 - dones_t) * target_q

        q = self.critic(states_t, actions_t)
        critic_loss = nn.functional.mse_loss(q, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_actions = self.actor(states_t)
        actor_loss = -self.critic(states_t, actor_actions).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, checkpoint_path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
