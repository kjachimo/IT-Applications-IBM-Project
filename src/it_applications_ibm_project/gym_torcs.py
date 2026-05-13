import copy
from typing import Any, Mapping, Sequence

import numpy as np
from gym import spaces

from it_applications_ibm_project.driver_action import ActionData, DriverAction
from it_applications_ibm_project.server_state import SensorData, ServerState
from it_applications_ibm_project.simple_client import SimpleClient
from it_applications_ibm_project import torcs_utils


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

    def step(self, action: ActionData) -> tuple[SensorData, float, int, dict]:
        client = self.client
        driver_action: DriverAction = client.R
        server_state: ServerState = client.S

        driver_action.d = action

        obs_pre = copy.deepcopy(server_state)

        client.respond_to_server()
        client.get_servers_input()

        obs = server_state

        self.observation = server_state.d

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

        if driver_action.d.get("meta", 0) is True:  # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.observation, reward, driver_action.d.get("meta", 0), {}

    def reset(self, relaunch: bool = False) -> SensorData:
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

        self.observation = (
            client.S.d
        )  # Make an obsevation from a raw observation vector from TORCS

        self.last_u = None

        self.initial_reset = False
        return self.observation

    def end(self) -> None:
        torcs_utils.stop_torcs(self._torcs_process)

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

    def make_observaton(self, raw_obs: ServerState) -> SensorData:
        obs_data: SensorData = raw_obs.d
        # if self.vision:
        #     image_rgb = self.obs_vision_to_image_rgb(obs_data["vision"])
        # else:
        image_rgb = None

        focus = np.array(obs_data["focus"], dtype=np.float32) / 200.0
        angle = obs_data["angle"]
        speed_x = obs_data["speedX"] / self.default_speed
        speed_y = obs_data["speedY"] / self.default_speed
        speed_z = obs_data["speedZ"] / self.default_speed
        rpm = obs_data["rpm"]
        track = np.array(obs_data["track"], dtype=np.float32) / 200.0
        track_pos = obs_data["trackPos"]
        gear = obs_data["gear"]
        stucktimer = obs_data["stucktimer"]
        damage = obs_data["damage"]
        fuel = obs_data["fuel"]
        dist_raced = obs_data["distRaced"]
        dist_from_start = obs_data["distFromStart"]
        z_pos = obs_data["z"]
        wheel_spin_vel = np.array(obs_data["wheelSpinVel"], dtype=np.float32)
        opponents = obs_data["opponents"]

        return {
            "focus": focus,
            "angle": angle,
            "speedX": speed_x,
            "speedY": speed_y,
            "speedZ": speed_z,
            "rpm": rpm,
            "track": track,
            "trackPos": track_pos,
            "gear": gear,
            "stucktimer": stucktimer,
            "damage": damage,
            "fuel": fuel,
            "distRaced": dist_raced,
            "distFromStart": dist_from_start,
            "z": z_pos,
            "wheelSpinVel": wheel_spin_vel,
            "opponents": opponents,
            "vision": image_rgb,
        }
