import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

from ValueModel import ValueModel
from PolicyModel import PolicyModel

gamma = 0.99
learning_rate_policy = 1e-4
learning_rate_value = 1e-3
buffer_size = 100000
batch_size = 64
epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def collect_experience(env, policy_model, buffer, episodes=100):
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action_probs = policy_model(state_tensor)
            action = torch.multinomial(action_probs, num_samples=1).item()
            next_state, reward, done, _, _ = env.step(action)

            buffer.append((state, action, reward, next_state, done))
            if len(buffer) > buffer_size:
                buffer.popleft()

            state = next_state

def train(env, policy_model, value_model, optimizer_policy, optimizer_value, buffer, gamma, device):
    criterion_value = nn.MSELoss()
    policy_losses, value_losses = [], []

    for epoch in range(epochs):
        if len(buffer) < batch_size:
            continue

        batch = random.sample(buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        current_values = value_model(states_tensor)
        with torch.no_grad():
            next_values = value_model(next_states_tensor)
            target_values = rewards_tensor + gamma * next_values * (1 - dones_tensor)

        # Perdita Value Model
        value_loss = criterion_value(current_values, target_values)
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        # Aggiornamento Policy Model
        action_probs = policy_model(states_tensor)
        action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)))

        with torch.no_grad():
            advantages = target_values - current_values

        policy_loss = -(action_log_probs * advantages).mean()
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

    return policy_losses, value_losses

def main(): 
    env = gym.make("Acrobot-v1")

    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_model = PolicyModel(input_dim=input_dim, action_dim=action_dim).to(device)
    value_model = ValueModel(input_dim=input_dim).to(device)

    optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate_policy)
    optimizer_value = optim.Adam(value_model.parameters(), lr=learning_rate_value)

    buffer = deque(maxlen=buffer_size)

    print("Collecting experience...")
    collect_experience(env, policy_model, buffer)

    print("Training models...")
    policy_losses, value_losses = train(
        env, policy_model, value_model, optimizer_policy, optimizer_value, buffer, gamma, device
    )

    print("Training complete. Saving models...")
    torch.save(policy_model.state_dict(), "policy_model.pth")
    torch.save(value_model.state_dict(), "value_model.pth")

if __name__ == "__main__":
    main()
