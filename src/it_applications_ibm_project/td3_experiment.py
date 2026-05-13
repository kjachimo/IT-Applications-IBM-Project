from it_applications_ibm_project.gym_torcs import TorcsEnv
from it_applications_ibm_project.td3_agent import TD3, ReplayBuffer
from it_applications_ibm_project.torcs_utils import automatic_transmission
from it_applications_ibm_project.driver_action import DriverAction
from it_applications_ibm_project.expert import drive_modular
import numpy as np


def main() -> None:
    vision = False
    episode_count = 10
    max_steps = 5000
    reward = 0
    done = False
    step = 0

    env = TorcsEnv(vision=vision)

    model = TD3(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=1,
    )
    replay_buffer = ReplayBuffer(
        state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]
    )

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
            if j == 0:
                expert_action = action.copy()
            action["gear"] = automatic_transmission(ob)
            driver_action = DriverAction()
            driver_action.d = action
            driver_action.clip_to_limits()
            action = driver_action.d

            next_ob, reward, done, _ = env.step(action)

            expert_action = drive_modular(ob, expert_action)

            replay_buffer.add(ob, expert_action, next_ob, reward, done)

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

        model.save(f"td3_actor_episode_{i}.pth")

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    main()
