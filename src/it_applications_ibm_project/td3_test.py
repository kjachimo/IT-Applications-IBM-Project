from it_applications_ibm_project.gym_torcs import TorcsEnv
from it_applications_ibm_project.td3_agent import TD3, ReplayBuffer
from it_applications_ibm_project.torcs_utils import automatic_transmission
from it_applications_ibm_project.driver_action import DriverAction
from it_applications_ibm_project.expert import drive_modular
import numpy as np
from pathlib import Path
import sys

SAVE_DIR = Path("models/td3")
LAST_MODEL_PATH = SAVE_DIR / "td3_actor_last.pth"
BEST_MODEL_PATH = SAVE_DIR / "td3_actor_best.pth"


def main() -> None:
    vision = False
    episode_count = 100
    max_steps = 5000
    done = False
    step = 0

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    env = TorcsEnv(vision=vision)

    model = TD3(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=1,
    )
    if len(sys.argv) > 1 and sys.argv[1] == "best":
        model.load(str(BEST_MODEL_PATH))
    else:
        model.load(str(LAST_MODEL_PATH))

    print("TORCS Experiment Start.")
    for i in range(episode_count):
        print("Episode : " + str(i))

        if np.mod(i, 5) == 0:
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        total_reward = 0.0
        for j in range(max_steps):
            action = model.select_action(ob)
            action["gear"] = automatic_transmission(ob)
            driver_action = DriverAction()
            driver_action.d = action
            driver_action.clip_to_limits()
            action = driver_action.d

            next_ob, reward, done, _ = env.step(action)

            ob = next_ob
            total_reward += reward

            step += 1
            if done:
                break

        print("TOTAL REWARD @ " + str(i) + " -th Episode  :  " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    main()
