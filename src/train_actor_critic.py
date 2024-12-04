import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

# Import custom models
from TransitionModel import TransitionModel  
from RewardModel import RewardModel          
from PolicyModel import PolicyModel          
from ValueModel import ValueModel
from DiscountModel import DiscountModel

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
state_dim = 6
action_dim = 3
num_episodes = 50
imagination_horizon = 5
buffer_capacity = 200
batch_size = 128
gamma = 0.99
lambda_ = 0.95
learning_rate = 2e-5

# Initialize models
transition_model = TransitionModel(state_dim, action_dim).to(device)
reward_model = RewardModel(state_dim).to(device)
policy_model = PolicyModel(input_dim=state_dim, action_dim=action_dim).to(device)
value_model = ValueModel(input_dim=state_dim).to(device)
discount_model = DiscountModel(input_dim=state_dim).to(device)

policy_optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
value_optimizer = optim.Adam(value_model.parameters(), lr=learning_rate)
discount_optimizer = optim.Adam(discount_model.parameters(), lr=learning_rate)

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
        return torch.tensor(np.array([item for item in data]), dtype=torch.float32).to(device)

    return preprocess(states), preprocess(actions).long(), preprocess(rewards), preprocess(next_states), preprocess(dones)

# Compute V_lambda with discount factors and horizon
def compute_v_lambda_with_discount(rewards, values, discounts, gamma=0.99, lambda_=0.95, horizon=15):
    batch_size = rewards.size(0)
    v_lambdas = torch.zeros_like(rewards, device=device)
    
    for i in range(batch_size):
        H = min(horizon, rewards.size(1))  # Use horizon or sequence length, whichever is smaller
        gae = 0.0
        next_value = values[i, -1]  # Last value for the current trajectory
        for t in reversed(range(H)):
            delta = rewards[i, t] + gamma * discounts[i, t] * next_value - values[i, t]
            gae = delta + gamma * lambda_ * gae
            v_lambdas[i, t] = gae + values[i, t]
            next_value = values[i, t]
    return v_lambdas

# Training loop
def train_models(policy_model, value_model, discount_model, buffer, num_episodes=100, imagination_horizon=15):
    max_grad_norm = 10
    factor_entropy = 0.3
    value_losses, policy_losses, discount_losses = [], [], []

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

        print(f"Collect new experience, dim_buffer: {len(buffer)}")
        # Train if buffer has enough experiences
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = sample_batch(buffer, batch_size)

            # Predict discounts
            predicted_discounts = discount_model(states).squeeze()
            discount_loss = torch.nn.functional.mse_loss(predicted_discounts, 1 - dones)
            discount_losses.append(discount_loss.item())

            discount_optimizer.zero_grad()
            discount_loss.backward()
            discount_optimizer.step()

            # Critic update
            values = value_model(states).squeeze()
            next_values = value_model(next_states).squeeze()
            td_targets = rewards + gamma * predicted_discounts * next_values * (1 - dones)
            value_loss = ((values - td_targets.detach()) ** 2).mean()
            value_losses.append(value_loss.item())
            
            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_grad_norm)
            value_optimizer.step()

            # Actor update
            advantages = compute_v_lambda_with_discount(
                rewards.unsqueeze(-1), 
                values.unsqueeze(-1), 
                predicted_discounts.unsqueeze(-1), 
                gamma, 
                lambda_, 
                imagination_horizon
            ) - values

            action_probs = policy_model(states)
            action_distribution = torch.distributions.Categorical(action_probs)
            log_probs = action_distribution.log_prob(actions)
            entropy = action_distribution.entropy().mean()
            policy_loss = -(log_probs * advantages.detach()).mean() - factor_entropy * entropy
            policy_losses.append(policy_loss.item())
            
            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            policy_optimizer.step()

        if episode % 1 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Value Loss: {value_loss.item():.4f}, "
                  f"Policy Loss: {policy_loss.item():.4f}, "
                  f"Discount Loss: {discount_loss.item():.4f}")

    return value_losses, policy_losses, discount_losses

# Save models
def save_models(policy_model, value_model, discount_model, policy_path="../models/policy_model.pth", 
                value_path="../models/value_model.pth", discount_path="../models/discount_model.pth"):
    
    torch.save(policy_model.state_dict(), policy_path)
    torch.save(value_model.state_dict(), value_path)
    torch.save(discount_model.state_dict(), discount_path)
    print(f"Models saved:\n - Policy Model: {policy_path}\n - Value Model: {value_path}\n - Discount Model: {discount_path}")

# Main function
if __name__ == "__main__":
    print("Starting training...")
    value_losses, policy_losses, discount_losses = train_models(
        policy_model, value_model, discount_model, buffer, num_episodes=num_episodes, imagination_horizon=imagination_horizon
    )

    # Save models
    save_models(policy_model, value_model, discount_model)
    
    env.close()
