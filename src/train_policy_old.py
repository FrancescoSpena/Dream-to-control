import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from Policy import PolicyModel
from RewardModel import RewardModel
from TransitionModel import TransitionModel
from ValueModel import ValueModel


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def generate_experience(state, action, transition_model, reward_model, device):
    next_state = transition_model(state, action)
    reward = reward_model(state)
    done = torch.tensor([1.0 if next_state[0, 0].item() > 1.0 else 0.0], device=device)  # Esempio basato sul primo elemento
    return next_state, reward, done


def compute_advantages(rewards, values, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = returns - torch.tensor(values, dtype=torch.float32)
    return advantages


def train_policy(env, policy_model, value_model, transition_model, reward_model, optimizer, buffer, batch_size=64, gamma=0.99, device='cpu'):
    value_model.eval()
    transition_model.eval()
    reward_model.eval()

    for episode in range(1000):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)

        rewards = []
        log_probs = []
        values = []

        done = False
        while not done:
            action_probs = policy_model(state.unsqueeze(0))  
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            one_hot_action = torch.nn.functional.one_hot(action, num_classes=3).float().to(device)


            next_state, reward, done = generate_experience(state.unsqueeze(0), one_hot_action, transition_model, reward_model, device)
            buffer.add((state, action.item(), reward.item(), next_state, done.item()))

            log_probs.append(action_distribution.log_prob(action))
            rewards.append(reward.item())
            with torch.no_grad():
                values.append(value_model(state.unsqueeze(0)).item())

            state = next_state.squeeze(0)  


        advantages = compute_advantages(rewards, values, gamma).to(device)
        policy_loss = -torch.sum(torch.stack(log_probs) * advantages.detach())

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}, Reward: {sum(rewards):.2f}, Policy Loss: {policy_loss.item():.4f}")


def main():
    env_name = "Acrobot-v1"
    gamma = 0.99
    learning_rate_policy = 1e-3
    hidden_units_policy = [128, 128]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    value_model = ValueModel(input_dim=state_dim).to(device)
    value_model.load_state_dict(torch.load("value_model.pth"))

    transition_model = TransitionModel(state_dim=state_dim, action_dim=action_dim).to(device)
    transition_model.load_state_dict(torch.load("transition_model.pth"))

    reward_model = RewardModel(state_dim=state_dim).to(device)
    reward_model.load_state_dict(torch.load("reward_model.pth"))

    policy_model = PolicyModel(input_dim=state_dim, action_dim=action_dim, hidden_units=hidden_units_policy).to(device)

    buffer = ReplayBuffer()

    optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate_policy)

    print("Training policy model...")
    train_policy(env, policy_model, value_model, transition_model, reward_model, optimizer_policy, buffer, gamma=gamma, device=device)

    torch.save(policy_model.state_dict(), "policy_model.pth")
    print("Training complete. Policy model saved.")


if __name__ == "__main__":
    main()
