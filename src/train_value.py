import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from ValueModel import ValueModel
from Dataset import ValueDataset


def collect_value_data(env, episodes=100, max_steps=500, gamma=0.99):
    states = []
    target_values = []
    for episode in range(episodes):
        episode_states = []
        episode_rewards = []
        state, _ = env.reset()
        for _ in range(max_steps):
            action = env.action_space.sample()  
            next_state, reward, done, _, _ = env.step(action)
            episode_states.append(state)
            episode_rewards.append(reward)
            state = next_state
            if done:
                break

        G = 0
        for reward in reversed(episode_rewards):
            G = reward + gamma * G
            target_values.insert(0, G)  

        states.extend(episode_states)

    return np.array(states, dtype=np.float32), np.array(target_values, dtype=np.float32)


def train_value_model(model, dataloader, optimizer, criterion, epochs=50, patience=10, device='cpu'):
    model.to(device)
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for states, target_values in dataloader:
            states = states.to(device)
            target_values = target_values.to(device)

            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, target_values)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "value_model.pth")  
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


def main():
    env_name = "Acrobot-v1"
    episodes = 200
    batch_size = 32
    learning_rate = 3e-4
    epochs = 100
    patience = 10  # Early stopping patience
    gamma = 0.99  # Fattore di sconto

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = gym.make(env_name)

    print("Collecting value data...")
    states, target_values = collect_value_data(env, episodes=episodes, gamma=gamma)

    dataset = ValueDataset(states, target_values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Initializing value model...")
    input_dim = states.shape[1]
    model = ValueModel(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #criterion = nn.MSELoss()  
    criterion = nn.SmoothL1Loss()


    print("Training value model...")
    train_value_model(model, dataloader, optimizer, criterion, epochs=epochs, patience=patience, device=device)
    print("Training complete. Best model saved as 'value_model.pth'")


if __name__ == "__main__":
    main()
