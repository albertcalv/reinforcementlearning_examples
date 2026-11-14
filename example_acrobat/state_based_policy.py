
import os
import math
import gymnasium as gym

from utils.plot_generation import save_animation, save_frame_grid

# Set up the output directory for images
PATH_IMG = "example_acrobat/img"
os.makedirs(PATH_IMG, exist_ok=True)

# Environment
env = gym.make("Acrobot-v1", render_mode="rgb_array")
env.unwrapped.book_or_nips = "nips"
env.action_space.seed(123)

# Step, Reward and Termination
MAX_STEPS = 100  
EPISODES = 1  
total_reward = 0
success = False

for episode in range(EPISODES):

    observation, info = env.reset(
        seed=123 + episode,
        options={"low": -0.2, "high": 0.2})
    episode_reward = 0
    
    frames = []
    for step in range(MAX_STEPS):
        frames.append(env.render())

        # Decode joint angles and angular velocities from the observation.
        theta1 = math.atan2(observation[1], observation[0])
        theta2 = math.atan2(observation[3], observation[2])
        theta1_dot = observation[4]
        theta2_dot = observation[5]

        # Simple energy signal heuristic 
        energy_signal = theta1_dot + 2 * theta2_dot

        action = 2 if energy_signal > 0 else 0
        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        total_reward += reward

        print(
            f"[Episode {episode + 1:02d}, step {step + 1:03d}]: "
            f"action={action}, reward={reward:.3f}, "
            f"terminated={terminated}, truncated={truncated}")

        if terminated or truncated:
            success = success or terminated
            break

    print(
        f"[Episode {episode + 1:02d}] finished after {step + 1} steps "
        f"with reward {episode_reward:.3f} and success={success}\n"
        + "-" * 20)

    gif_path = os.path.join(
        PATH_IMG, f"state_based_policy_episode_{episode + 1}.gif"
    )
    save_animation(frames, gif_path)

    grid_path = os.path.join(
        PATH_IMG, f"state_based_policy_episode_{episode + 1}.png"
    )
    save_frame_grid(frames, grid_path)

print(f"Total reward: {total_reward:.3f}")
print(f"Success: {success}")

env.close()