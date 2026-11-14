import os
import gymnasium as gym

from utils.plot_generation import save_animation, save_frame_grid

# Output directory
PATH_IMG = "example_acrobat/img"
os.makedirs(PATH_IMG, exist_ok=True)

# Environment
env = gym.make("Acrobot-v1", render_mode="rgb_array")
env.unwrapped.book_or_nips = "nips"
env.action_space.seed(123)

MAX_STEPS = 100
EPISODES = 1

success = False
total_reward = 0

for episode in range(EPISODES):

    # Reset environment
    observation, info = env.reset(
        seed=123 + episode,
        options={"low": -0.2, "high": 0.2}
    )

    episode_reward = 0
    frames = []  # Reset frames for each episode

    for step in range(MAX_STEPS):

        # Capture the current image
        frame = env.render()
        frames.append(frame)

        # Random action
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

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

    # Save GIF animation and frame grid.
    gif_path = os.path.join(
        PATH_IMG, f"acrobot_episode_{episode + 1}.gif"
    )
    save_animation(frames, gif_path)

    grid_path = os.path.join(
        PATH_IMG, f"random_policy_episode_{episode + 1}.png"
    )
    save_frame_grid(frames, grid_path)

env.close()
