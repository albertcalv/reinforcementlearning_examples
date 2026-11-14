import os

import gymnasium as gym
import torch
import torch.nn as nn

from utils.plot_generation import save_animation

PATH_IMG = "example_acrobat/img"
MODEL_PATH = "example_acrobat/dqn/dqn_acrobot.pt"
GIF_PATH = os.path.join(PATH_IMG, "dqn_inference.gif")
os.makedirs(PATH_IMG, exist_ok=True)

EPISODES = 1
MAX_STEPS = 1000
SEED = 123


class QNetwork(nn.Module):
    """MLP for Q-value prediction."""
    def __init__(self, n_obs: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions))

    def forward(self, x):
        return self.net(x)


env = gym.make("Acrobot-v1", render_mode="rgb_array", max_episode_steps=MAX_STEPS)
policy_net = QNetwork(env.observation_space.shape[0], env.action_space.n)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
policy_net.eval()

frames = []
for episode in range(EPISODES):
    observation, _ = env.reset(seed=SEED + episode)
    episode_reward = 0.0

    for step in range(MAX_STEPS):
        frames.append(env.render())

        with torch.no_grad():
            q_values = policy_net(torch.as_tensor(observation, dtype=torch.float32))
        action = int(q_values.argmax().item())

        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            break

    print(
        f"[Episode {episode + 1:02d}] finished after {step + 1} steps "
        f"with reward {episode_reward:.1f}, success={terminated}")

env.close()
save_animation(frames, GIF_PATH)
