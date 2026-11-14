import os
import gymnasium as gym
from utils.plot_generation import save_animation, save_frame_grid


PATH_IMG = "example_mountain_car/img"
os.makedirs(PATH_IMG, exist_ok=True)

env = gym.make(
    "MountainCar-v0",
    render_mode="rgb_array",
    goal_velocity=0.05,
)

MAX_STEPS = 200
EPISODES = 1

success = False
total_reward = 0


for episode in range(EPISODES):
    observation, info = env.reset(seed=123 + episode)
    episode_reward = 0
    frames = []

    for step in range(MAX_STEPS):
        frames.append(env.render())

        # Small heuristic policy
        position, velocity = observation

        if velocity > 0:
            action = 2  # Push right
        elif velocity < 0:
            action = 0  # Push left
        else:
            action = 2 if position > -0.5 else 0

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        episode_reward += reward
        total_reward += reward

        print(
            f"[Episode {episode + 1:02d}, "
            f"step {step + 1:03d}]: "
            f"action={action}, reward={reward:.3f}, "
            f"terminated={terminated}, truncated={truncated}"
        )

        if terminated or truncated:
            success = success or terminated
            break

    print(
        f"Episode {episode + 1} finished after {step + 1} steps. "
        f"Reward: {episode_reward:.3f}, success={success}"
    )

    gif_path = os.path.join(
        PATH_IMG, f"policy_mountain_car_episode_{episode + 1}.gif"
    )
    save_animation(frames, gif_path)

    """
    grid_path = os.path.join(
        PATH_IMG, f"heuristic_policy_episode_{episode + 1}.png")
    save_frame_grid(frames, grid_path)
    """

env.close()

print(f"Total reward: {total_reward:.3f}")
print(f"Success: {success}")
