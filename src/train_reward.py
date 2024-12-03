import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from RewardModel import RewardModel
from Dataset import RewardDataset

torch.cuda.empty_cache()

def collect_data(env, steps=10000, max_steps=500):
    transitions = []
    state, _ = env.reset()  
    state_dim = env.observation_space.shape[0]
    step_count = 0

    while len(transitions) < steps:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if state.shape != (state_dim,):
            continue

        transitions.append((state, reward))
        step_count += 1
        state = next_state

        if terminated or truncated or step_count >= max_steps:
            state, _ = env.reset()
            step_count = 0

    return transitions

def collect_balanced_data(env, steps_per_bin, state_dim, reward_bins):
    transitions = {bin_idx: [] for bin_idx in range(len(reward_bins) - 1)}
    total_steps = steps_per_bin * (len(reward_bins) - 1)
    state, _ = env.reset()
    step_count = 0

    while sum(len(v) for v in transitions.values()) < total_steps:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if state.shape != (state_dim,):
            state, _ = env.reset()  
            continue

        # Bin the reward
        bin_idx = np.digitize(reward, reward_bins) - 1
        if 0 <= bin_idx < len(reward_bins) - 1:
            if len(transitions[bin_idx]) < steps_per_bin:
                transitions[bin_idx].append((state, reward))

        state = next_state
        step_count += 1

        if terminated or truncated or step_count >= 100:
            state, _ = env.reset()
            step_count = 0

    balanced_transitions = [
        transition for bin_transitions in transitions.values() for transition in bin_transitions
    ]

    return balanced_transitions


def check_balance_rewards(transitions, reward_bins):
    bin_counts = Counter()
    for _, reward in transitions:
        bin_idx = np.digitize(reward, reward_bins) - 1
        bin_counts[bin_idx] += 1

    for bin_idx in range(len(reward_bins) - 1):
        print(f"Bin {bin_idx} [{reward_bins[bin_idx]}, {reward_bins[bin_idx + 1]}]: {bin_counts[bin_idx]} samples")
    
    return bin_counts


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=50, patience=10, device='cpu'):
    model.to(device)
    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_value = []
    validation_value = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for states, rewards in train_loader:
            states = states.to(device)
            rewards = rewards.to(device)

            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, rewards)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for states, rewards in val_loader:
                states = states.to(device)
                rewards = rewards.to(device)
                outputs = model(states)
                loss = criterion(outputs, rewards)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        train_value.append(avg_train_loss)
        validation_value.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_value, validation_value

def plot_loss_with_average(loss_values, epochs):
    if len(loss_values) != epochs:
        raise ValueError("loss_values different from number of epochs.")

    epoch_numbers = list(range(1, epochs + 1))
    cumulative_average = np.cumsum(loss_values) / np.arange(1, len(loss_values) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, loss_values, label='Loss', color='blue', alpha=0.6)
    plt.plot(epoch_numbers, cumulative_average, label='Average Loss', color='orange', linewidth=2)
    
    plt.title('Loss and Average Loss Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Configurazioni
    env_name = "Acrobot-v1"
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-4
    hidden_units = [32, 32]
    epochs = 50
    patience = 5 

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]

    print("Collecting data...")
    transitions = collect_data(env)
    
    dataset = RewardDataset(transitions)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing model...")
    model = RewardModel(state_dim=state_dim, hidden_units=hidden_units)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    print("Training model...")
    train_value, valid_value = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=epochs, patience=patience, device='cuda' if torch.cuda.is_available() else 'cpu')

    plot_loss_with_average(train_value,epochs)
    plot_loss_with_average(valid_value,epochs)
    
    torch.save(model.state_dict(), "models/reward_model.pth")
    print("Training complete. Best model saved as 'reward_model.pth'")


if __name__ == "__main__":
    main()
