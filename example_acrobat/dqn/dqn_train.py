
import os
import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Output directories
PATH_IMG = "example_acrobat/img"
PATH_MODEL = "example_acrobat/dqn"

os.makedirs(PATH_IMG, exist_ok=True)
os.makedirs(PATH_MODEL, exist_ok=True)

EPISODES = 1500
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 50_000
MIN_BUFFER = 1_000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

TARGET_UPDATE_STEPS = 2000
SEED = 123
EVAL_EPISODES = 100

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = gym.make("Acrobot-v1")
env.action_space.seed(SEED)

n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n


class QNetwork(nn.Module):
    """MLP for Q-value prediction."""
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions))

    def forward(self, x):
        return self.net(x)


policy_net = QNetwork(n_obs, n_actions)
target_net = QNetwork(n_obs, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = deque(maxlen=BUFFER_SIZE)


def select_action(state, epsilon):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return env.action_space.sample()

    with torch.no_grad():
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32
        )
        q_values = policy_net(state_tensor)
        return int(q_values.argmax().item())


def optimize():
    """Perform one Double DQN gradient update."""
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.as_tensor(
        np.array(states), dtype=torch.float32
    )
    actions = torch.as_tensor(
        actions, dtype=torch.int64
    ).unsqueeze(1)
    rewards = torch.as_tensor(
        rewards, dtype=torch.float32
    )
    next_states = torch.as_tensor(
        np.array(next_states), dtype=torch.float32
    )
    dones = torch.as_tensor(
        dones, dtype=torch.float32
    )

    # Q-values for the actions actually taken
    q = policy_net(states).gather(
        1, actions
    ).squeeze(1)

    # Double DQN:
    # Policy network selects the next action.
    # Target network evaluates that action.
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(
            dim=1, keepdim=True
        )

        next_q = target_net(next_states).gather(
            1, next_actions
        ).squeeze(1)

        target = rewards + GAMMA * next_q * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(q, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
        policy_net.parameters(), 10.0
    )
    optimizer.step()


# Training
epsilon = EPS_START
total_steps = 0
episode_rewards = []

for episode in range(EPISODES):
    state, _ = env.reset(seed=SEED + episode)
    episode_reward = 0.0
    done = False

    while not done:
        action = select_action(state, epsilon)

        next_state, reward, terminated, truncated, _ = (
            env.step(action)
        )

        done = terminated or truncated
        
        buffer.append((
            state,
            action,
            reward,
            next_state,
            float(terminated),
        ))

        state = next_state
        episode_reward += reward
        total_steps += 1

        if len(buffer) >= MIN_BUFFER:
            optimize()

        if total_steps % TARGET_UPDATE_STEPS == 0:
            target_net.load_state_dict(
                policy_net.state_dict()
            )

    # Gradually reduce exploration
    epsilon = max(
        EPS_END, epsilon * EPS_DECAY
    )

    episode_rewards.append(episode_reward)

    if (episode + 1) % 10 == 0:
        avg = np.mean(episode_rewards[-10:])
        print(
            f"Episode {episode + 1:04d} | "
            f"Average reward (10): {avg:.1f} | "
            f"Epsilon: {epsilon:.3f}"
        )

env.close()

# Save trained weights
model_path = os.path.join(
    PATH_MODEL, "dqn_acrobot.pt"
)
torch.save(policy_net.state_dict(), model_path)
print(f"\nModel saved to: {model_path}")

# Plot training curve
window = 10
smoothed = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid",
)

plt.figure(figsize=(8, 4))
plt.plot(
    episode_rewards,
    alpha=0.3,
    label="Episode reward",
)
plt.plot(
    range(window - 1, len(episode_rewards)),
    smoothed,
    label=f"Moving avg ({window})",
)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Double DQN on Acrobot-v1")
plt.legend()
plt.tight_layout()

plot_path = os.path.join(
    PATH_IMG, "dqn_training_curve.png"
)
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Training curve saved to: {plot_path}")
