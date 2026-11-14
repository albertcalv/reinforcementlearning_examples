import os

import gymnasium as gym
import torch
import torch.nn as nn

from utils.plot_generation import save_animation

PATH_IMG = "example_acrobat/img"
MODEL_PATH = "example_acrobat/reinforce/reinforce_acrobot.pt"
GIF_PATH = os.path.join(PATH_IMG, "reinforce_inference.gif")
os.makedirs(PATH_IMG, exist_ok=True)

MAX_STEPS = 500
SEED = 123


class PolicyNetwork(nn.Module):
    """Same architecture used in reinforce_train.py."""

    def __init__(self, n_obs: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


env = gym.make("Acrobot-v1", render_mode="rgb_array", max_episode_steps=MAX_STEPS)
policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
policy.eval()

observation, _ = env.reset(seed=SEED)
frames = []
episode_reward = 0.0

for step in range(MAX_STEPS):
    frames.append(env.render())

    # Greedy action: most probable one
    with torch.no_grad():
        logits = policy(torch.as_tensor(observation, dtype=torch.float32))
    action = int(logits.argmax().item())

    observation, reward, terminated, truncated, _ = env.step(action)
    episode_reward += reward
    if terminated or truncated:
        break

print(
    f"Finished after {step + 1} steps with reward {episode_reward:.1f}, "
    f"success={terminated}"
)
env.close()
save_animation(frames, GIF_PATH)
