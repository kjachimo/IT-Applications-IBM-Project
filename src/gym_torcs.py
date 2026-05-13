import copy

import numpy as np
from gym import spaces

from driver_action import DriverAction
from server_state import ServerState
from simple_client import SimpleClient
import torcs_utils


class TorcsEnv:
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = (
        5  # [km/h], episode terminates if car is running slower than this limit
    )
    default_speed = 50

    initial_reset = True

    def __init__(self, vision: bool = False):
        # print("Init")
        self.vision = vision

        self.initial_run = True
        self._torcs_process = None

        self.reset_torcs()

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        if vision is False:
            high = np.array([1.0, np.inf, np.inf, np.inf, 1.0, np.inf, 1.0, np.inf])
            low = np.array([0.0, -np.inf, -np.inf, -np.inf, 0.0, -np.inf, 0.0, -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array(
                [1.0, np.inf, np.inf, np.inf, 1.0, np.inf, 1.0, np.inf, 255]
            )
            low = np.array(
                [0.0, -np.inf, -np.inf, -np.inf, 0.0, -np.inf, 0.0, -np.inf, 0]
            )
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, action) -> tuple[tuple, float, bool, dict]:
        client = self.client
        driver_action = client.R
        server_state = client.S

        self.agent_to_torcs(action, driver_action)

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(server_state)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = server_state

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs.d["track"])
        sp = np.array(obs.d["speedX"])
        progress = sp * np.cos(obs.d["angle"])
        reward = progress

        # collision detection
        if obs.d["damage"] - obs_pre.d["damage"] > 0:
            reward = -1

        # Termination judgement #########################
        episode_terminate = False
        if track.min() < 0:  # Episode is terminated if the car is out of track
            reward = -1
            episode_terminate = True
            driver_action.d["meta"] = True

        if (
            self.terminal_judge_start < self.time_step
        ):  # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                episode_terminate = True
                driver_action.d["meta"] = True

        if (
            np.cos(obs.d["angle"]) < 0
        ):  # Episode is terminated if the agent runs backward
            episode_terminate = True
            driver_action.d["meta"] = True

        if driver_action.d["meta"] is True:  # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, driver_action.d["meta"], {}

    def reset(self, relaunch: bool = False) -> tuple:
        # print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = SimpleClient(vision=self.vision)  # Open new UDP in vtorcs

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self) -> None:
        torcs_utils.stop_torcs(self._torcs_process)

    def get_obs(self) -> tuple:
        return self.observation

    def reset_torcs(self) -> None:
        print("relaunching torcs")
        self._torcs_process = torcs_utils.reset_torcs(
            vision=self.vision,
            process=self._torcs_process,
            wait_after_launch=9.0,
            wait_after_autostart=0.5,
            kill_fallback=False,
            exe="wtorcs.exe",
            use_wine=True,
        )

    def agent_to_torcs(self, action, driver_action: DriverAction) -> None:
        steer, accel, brake = self._action_values(action)
        driver_action.d["steer"] = steer
        driver_action.d["accel"] = accel
        driver_action.d["brake"] = brake

    def _action_values(self, action) -> tuple[float, float, float]:
        if isinstance(action, dict):
            if "steer" in action:
                return (
                    float(action["steer"]),
                    float(action["accel"]),
                    float(action["brake"]),
                )
            return (
                float(action["steering"]),
                float(action["acceleration"]),
                float(action["brake"]),
            )
        if hasattr(action, "steering"):
            return (
                float(action.steering),
                float(action.acceleration),
                float(action.brake),
            )
        if hasattr(action, "steer"):
            return (
                float(action.steer),
                float(action.accel),
                float(action.brake),
            )
        try:
            steer, accel, brake = action
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Action must be mapping, object, or 3-item sequence"
            ) from exc
        return float(steer), float(accel), float(brake)

    def obs_vision_to_image_rgb(
        self, obs_image_vec: list[float] | np.ndarray
    ) -> np.ndarray:
        image_vec = obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0, 12286, 3):
            temp.append(image_vec[i])
            temp.append(image_vec[i + 1])
            temp.append(image_vec[i + 2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs: ServerState) -> tuple:
        if self.vision:
            image_rgb = self.obs_vision_to_image_rgb(raw_obs.d[names[8]])
        else:
            image_rgb = None

        focus = np.array(raw_obs.d["focus"], dtype=np.float32) / 200.0
        speed_x = np.array(raw_obs.d["speedX"], dtype=np.float32) / self.default_speed
        speed_y = np.array(raw_obs.d["speedY"], dtype=np.float32) / self.default_speed
        speed_z = np.array(raw_obs.d["speedZ"], dtype=np.float32) / self.default_speed
        opponents = np.array(raw_obs.d["opponents"], dtype=np.float32) / 200.0
        rpm = np.array(raw_obs.d["rpm"], dtype=np.float32)
        track = np.array(raw_obs.d["track"], dtype=np.float32) / 200.0
        wheel_spin_vel = np.array(raw_obs.d["wheelSpinVel"], dtype=np.float32)

        if self.vision:
            return (
                focus,
                speed_x,
                speed_y,
                speed_z,
                opponents,
                rpm,
                track,
                wheel_spin_vel,
                image_rgb,
            )

        return (
            focus,
            speed_x,
            speed_y,
            speed_z,
            opponents,
            rpm,
            track,
            wheel_spin_vel,
        )
