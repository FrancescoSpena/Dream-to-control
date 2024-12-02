import gymnasium as gym 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from torch.utils.data import DataLoader

from TransitionModel import TransitionModel
from Dataset import TransitionDataset

def collect_data(env, steps, state_dim, action_dim):
    transitions = []
    state, _ = env.reset()  # Estrai lo stato dalla tupla (state, info)
    for _ in range(steps):
        action = env.action_space.sample()  # Seleziona azioni casuali
        next_state, _, terminated, truncated, _ = env.step(action)

        # Controllo che lo stato e il prossimo stato siano array di dimensione corretta
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)

        if state.shape != (state_dim,) or next_state.shape != (state_dim,):
            continue  # Ignora stati non validi

        # One-hot encode dell'azione
        action_one_hot = np.zeros(action_dim)
        action_one_hot[action] = 1

        # Salva la transizione
        transitions.append((state, action_one_hot, next_state))
        state = next_state

        # Reset se l'episodio termina
        if terminated or truncated:
            state, _ = env.reset()  # Estrai lo stato dalla tupla
    return transitions

def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for state, action, next_state in dataloader:
            optimizer.zero_grad()

            predicted_next_state = model(state,action)

            loss = criterion(predicted_next_state,next_state)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    env = gym.make("Acrobot-v1")

    bath_size = 64
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 

    epochs = 100
    lr = 1e-4

    print("Collect data...")
    transitions = collect_data(env,10000,state_dim=state_dim,action_dim=action_dim)
    
    dataset = TransitionDataset(transitions)
    dataloader = DataLoader(dataset, batch_size=bath_size, shuffle=True)

    model = TransitionModel(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Starting training...")
    train(model, dataloader, optimizer, criterion, epochs)

    torch.save(model.state_dict(), "transition_model.pth")
    print("Modello saved as 'transition_model.pth'")

