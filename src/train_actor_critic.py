import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import gymnasium as gym
import random

# Import custom models (assumed to exist in your project)
from TransitionModel import TransitionModel  
from RewardModel import RewardModel          
from PolicyModel import PolicyModel          
from ValueModel import ValueModel            

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
state_dim = 6    
action_dim = 3   
num_episodes = 500
imagination_horizon = 70
buffer_capacity = 100000
batch_size = 64
gamma = 0.99
lambda_ = 0.95
learning_rate = 2e-5

# Initialize models
transition_model = TransitionModel(state_dim, action_dim).to(device)
reward_model = RewardModel(state_dim).to(device)
policy_model = PolicyModel(input_dim=state_dim, action_dim=action_dim).to(device)
value_model = ValueModel(input_dim=state_dim).to(device)

policy_optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
value_optimizer = optim.Adam(value_model.parameters(), lr=learning_rate)

# Experience replay buffer
buffer = deque(maxlen=buffer_capacity)

# Environment setup
env = gym.make('Acrobot-v1')

# Helper functions
def add_experience(buffer, state, action, reward, next_state, done):
    buffer.append((state, action, reward, next_state, done))

def sample_batch(buffer, batch_size):
    batch = random.sample(buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    def preprocess(data):
        # If data contains tensors, convert to NumPy and move to CPU
        return np.array([item.cpu().numpy() if isinstance(item, torch.Tensor) else item for item in data])

    return (
        preprocess(states),
        preprocess(actions),
        preprocess(rewards),
        preprocess(next_states),
        preprocess(dones),
    )

def compute_v_lambda(rewards, values, gamma=0.99, lambda_=0.95):
    H = len(rewards)
    v_lambda = torch.zeros(H, device=device)
    gae = 0.0
    next_value = values[-1]
    for t in reversed(range(H)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * gae
        v_lambda[t] = gae + values[t]
        next_value = values[t]
    return v_lambda

# Training loop
def train_models(policy_model, value_model, buffer, num_episodes=100, imagination_horizon=15):
    max_grad_norm = 10
    factor_entropy = 0.3
    value_losses, policy_losses = [], []

    for episode in range(num_episodes):
        # Collect experiences from the environment
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action_probs = policy_model(state_tensor)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample().item()
            next_state, reward, done, _, _ = env.step(action)
            
            add_experience(buffer, state, action, reward, next_state, done)
            state = next_state

        # Train if buffer has enough experiences
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = sample_batch(buffer, batch_size)

            # Convert lists of numpy arrays to tensors
            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            actions = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
            dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)
            
            # Critic update
            values = value_model(states).squeeze()
            next_values = value_model(next_states).squeeze()
            td_targets = rewards + gamma * next_values * (1 - dones)
            value_loss = ((values - td_targets.detach()) ** 2).mean()
            value_losses.append(value_loss.item())
            
            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_grad_norm)
            value_optimizer.step()

            # Actor update
            action_probs = policy_model(states)
            action_distribution = torch.distributions.Categorical(action_probs)
            log_probs = action_distribution.log_prob(actions)
            advantages = td_targets - values.detach()

            entropy = action_distribution.entropy().mean()
            policy_loss = -(log_probs * advantages.detach()).mean() - factor_entropy * entropy
            policy_losses.append(policy_loss.item())
            
            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            policy_optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Value Loss: {value_loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}")

    return value_losses, policy_losses

def save_models(policy_model, value_model, policy_path="../models/policy_model.pth", value_path="../models/value_model.pth"):
    """Save the policy and value models to the specified paths."""
    torch.save(policy_model.state_dict(), policy_path)
    torch.save(value_model.state_dict(), value_path)
    print(f"Models saved:\n - Policy Model: {policy_path}\n - Value Model: {value_path}")

# Main function
if __name__ == "__main__":
    print("Starting training with environment interaction...")
    value_losses, policy_losses = train_models(policy_model, value_model, buffer, num_episodes=num_episodes)
    save_models(policy_model=policy_model,value_model=value_model)
    env.close()
