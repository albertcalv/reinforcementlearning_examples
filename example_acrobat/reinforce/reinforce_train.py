import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Output directories
PATH_IMG = "example_acrobat/img"
PATH_MODEL = "example_acrobat/reinforce"
os.makedirs(PATH_IMG, exist_ok=True)
os.makedirs(PATH_MODEL, exist_ok=True)

EPISODES = 500
GAMMA = 0.99
LR = 1e-3
SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = gym.make("Acrobot-v1")
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n


class PolicyNetwork(nn.Module):
    """MLP mapping an observation to action logits (stochastic policy)."""

    def __init__(self, n_obs: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions))

    def forward(self, x):
        return self.net(x)


policy = PolicyNetwork(n_obs, n_actions)
optimizer = optim.Adam(policy.parameters(), lr=LR)


def select_action(state):
    """Sample an action from the policy and return it with its log-probability."""
    logits = policy(torch.as_tensor(state, dtype=torch.float32))
    dist = Categorical(logits=logits)
    action = dist.sample()
    return int(action.item()), dist.log_prob(action)


def compute_returns(rewards):
    """Discounted return G_t = r_t + gamma * G_{t+1}, normalized for lower variance."""
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + GAMMA * g
        returns.append(g)
    returns = torch.tensor(returns[::-1], dtype=torch.float32)
    return (returns - returns.mean()) / (returns.std() + 1e-8)


episode_rewards = []

for episode in range(EPISODES):
    state, _ = env.reset(seed=SEED + episode)
    log_probs, rewards = [], []
    done = False

    while not done:
        action, log_prob = select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        log_probs.append(log_prob)
        rewards.append(reward)

    # maximize E[log pi(a|s) * G_t]  ->  minimize -log pi * G_t
    returns = compute_returns(rewards)
    loss = -(torch.stack(log_probs) * returns).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_rewards.append(sum(rewards))

    print(
        f"[Episode {episode + 1:03d}] "
        f"avg reward (last 10)={np.mean(episode_rewards[-10:]):.1f}")

env.close()

# Save the trained weights
model_path = os.path.join(PATH_MODEL, "reinforce_acrobot.pt")
torch.save(policy.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# Training curve
window = 10
smoothed = (
    np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    if len(episode_rewards) >= window
    else []
)
plt.figure(figsize=(8, 4))
plt.plot(episode_rewards, alpha=0.3, label="Episode reward")
if len(smoothed):
    plt.plot(range(window - 1, len(episode_rewards)), smoothed, label=f"Moving avg ({window})")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("REINFORCE on Acrobot-v1")
plt.legend()
plt.tight_layout()
curve_path = os.path.join(PATH_IMG, "reinforce_training_curve.png")
plt.savefig(curve_path)
print(f"Training curve saved to: {curve_path}")
