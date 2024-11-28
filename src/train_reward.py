import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from RewardModel import RewardModel
from Dataset import RewardDataset

torch.cuda.empty_cache()


# Funzione per raccogliere dati dall'ambiente
def collect_data(env, episodes=100, max_steps=500):
    transitions = []
    for episode in range(episodes):
        state, _ = env.reset()
        for _ in range(max_steps):
            action = env.action_space.sample()  
            next_state, reward, done, _, _ = env.step(action)
            transitions.append((state, reward))
            state = next_state
            if done:
                break
    return transitions


# Funzione per addestrare il modello con early stopping
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=50, patience=10, device='cpu'):
    model.to(device)
    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0

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

        # Calcola la perdita media
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "reward_model.pth")  
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


def main():
    # Configurazioni
    env_name = "Acrobot-v1"
    episodes = 200
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-4
    hidden_units = [32, 32]
    epochs = 50
    patience = 5 

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]

    print("Collecting data...")
    transitions = collect_data(env, episodes=episodes)

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
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs=epochs, patience=patience, device='cuda' if torch.cuda.is_available() else 'cpu')

    print("Training complete. Best model saved as 'reward_model.pth'")


if __name__ == "__main__":
    main()
