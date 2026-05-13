import numpy as np
import matplotlib.pyplot as plt

from it_applications_ibm_project.driver_action import ActionData
from it_applications_ibm_project.server_state import SensorData
from it_applications_ibm_project.torcs_utils import automatic_transmission


class Agent(object):
    def __init__(self, dim_action):
        self.dim_action = dim_action

    def act(self, ob: SensorData, reward, done, vision_on) -> ActionData:
        # print("ACT!")

        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        # vision is given as a tensor with size of (64*64, 3) = (4096, 3) <-- rgb
        # and values are in [0, 255]
        action: ActionData = {
            "steer": np.tanh(np.random.randn()),
            "accel": 1,
            "brake": np.tanh(np.random.randn()),
        }

        action["gear"] = automatic_transmission(ob)

        return action
