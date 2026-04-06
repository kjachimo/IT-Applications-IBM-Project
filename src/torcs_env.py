import math
from typing import Dict, Tuple

import numpy as np

from .client import Client


class TorcsRLEnv:
    terminal_judge_start = 250
    termination_limit_progress = 5.0

    def __init__(
        self,
        process_manager,
        port: int = 3001,
        vision: bool = False,
        relaunch_every: int = 3,
        max_steps: int = 2000,
    ):
        self.process_manager = process_manager
        self.port = port
        self.vision = vision
        self.relaunch_every = relaunch_every
        self.max_steps = max_steps
        self.client = None
        self.episode_index = 0
        self.time_step = 0
        self.prev_obs = None

    @property
    def state_dim(self) -> int:
        return 29

    @property
    def action_dim(self) -> int:
        return 3

    def _obs_to_state(self, obs: Dict) -> np.ndarray:
        track = np.array(obs["track"], dtype=np.float32) / 200.0
        wheel = np.array(obs["wheelSpinVel"], dtype=np.float32) / 200.0
        state = np.hstack(
            [
                np.array([obs["angle"]], dtype=np.float32),
                track,
                np.array([obs["trackPos"]], dtype=np.float32),
                np.array(
                    [obs["speedX"], obs["speedY"], obs["speedZ"]], dtype=np.float32
                )
                / 300.0,
                wheel,
                np.array([obs["rpm"]], dtype=np.float32) / 10000.0,
            ]
        )
        return state

    def _auto_gear(self, speed_x: float) -> int:
        if speed_x > 170:
            return 6
        if speed_x > 140:
            return 5
        if speed_x > 110:
            return 4
        if speed_x > 80:
            return 3
        if speed_x > 50:
            return 2
        return 1

    def _compute_reward_done(self, obs: Dict, prev_obs: Dict) -> Tuple[float, bool, Dict]:
        speed_x = float(obs["speedX"])
        angle = float(obs["angle"])
        track_pos = float(obs["trackPos"])
        progress = speed_x * math.cos(angle)

        reward = progress - abs(speed_x * math.sin(angle)) - speed_x * abs(track_pos)
        reward = reward / 100.0

        damage_delta = float(obs["damage"]) - float(prev_obs["damage"])
        if damage_delta > 0:
            reward -= 1.0

        done = False
        off_track = min(obs["track"]) < 0 or abs(track_pos) > 1.0
        backward = math.cos(angle) < 0
        stalled = (
            self.time_step > self.terminal_judge_start
            and progress < self.termination_limit_progress
        )
        timeout = self.time_step >= self.max_steps

        if off_track or backward or stalled or timeout:
            done = True

        info = {
            "progress": progress,
            "off_track": off_track,
            "backward": backward,
            "stalled": stalled,
            "timeout": timeout,
            "damage_delta": damage_delta,
        }
        return reward, done, info

    def reset(self, relaunch: bool = False):
        if self.client is not None:
            self.client.R.d["meta"] = 1
            self.client.respond_to_server()
            self.client.shutdown()
            self.client = None

        if self.episode_index == 0 or relaunch:
            self.process_manager.hard_reset()

        self.client = Client(
            p=self.port,
            vision=self.vision,
            parse_command_line=False,
        )
        self.client.get_servers_input()
            
        self.time_step = 0
        self.prev_obs = dict(self.client.S.d)
        state = self._obs_to_state(self.client.S.d)
        self.episode_index += 1
        return state

    def step(self, action):
        if self.client is None:
            raise RuntimeError("reset() must be called before step().")

        act = np.asarray(action, dtype=np.float32)
        steer = float(np.clip(act[0], -1.0, 1.0))
        accel = float(np.clip(act[1], 0.0, 1.0))
        brake = float(np.clip(act[2], 0.0, 1.0))

        # Avoid simultaneous high throttle and brake during exploration.
        if accel > 0.3 and brake > 0.3:
            brake *= 0.2

        self.client.R.d["steer"] = steer
        self.client.R.d["accel"] = accel
        self.client.R.d["brake"] = brake
        self.client.R.d["gear"] = self._auto_gear(float(self.client.S.d["speedX"]))
        self.client.R.d["meta"] = 0

        obs_pre = dict(self.client.S.d)
        self.client.respond_to_server()
        self.client.get_servers_input()
        obs = dict(self.client.S.d)

        reward, done, info = self._compute_reward_done(obs, obs_pre)
        self.time_step += 1

        if done:
            self.client.R.d["meta"] = 1
            self.client.respond_to_server()

        self.prev_obs = obs
        next_state = self._obs_to_state(obs)
        return next_state, reward, done, info

    def close(self, stop_torcs: bool = False):
        if self.client is not None:
            self.client.shutdown()
            self.client = None
        if stop_torcs:
            self.process_manager.stop()
