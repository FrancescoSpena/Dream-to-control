import gymnasium as gym
import torch
import torch.nn as nn

from src.Dataset import RewardDataset
from src.RewardModel import RewardModel

# Raccogliere i dati di addestramento
def collect_reward_data(env, steps):
    print("Startint collection data...")
    transitions = []
    state, _ = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        transitions.append((state, reward))  # Salva stato e ricompensa
        state = next_state

        if terminated or truncated:
            state, _ = env.reset()
    
    print("Complete collection data...")
    return transitions

# Addestramento
def train_reward_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for state, reward in dataloader:
            optimizer.zero_grad()
            predicted_reward = model(state)
            loss = criterion(predicted_reward, reward)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

if __name__ == '__main__':
    pass