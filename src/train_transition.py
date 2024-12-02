import gymnasium as gym 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
import numpy as np 

from TransitionModel import TransitionModel
from Dataset import TransitionDataset

#hyperparam: 
STATE_DIM = 6
ACTION_DIM = 3 
HIDDEN_UNITS = [128,128]
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 100 
ENV_STEPS = 5000


def collect_data(env, steps):
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

        if state.shape != (STATE_DIM,) or next_state.shape != (STATE_DIM,):
            continue  # Ignora stati non validi

        # One-hot encode dell'azione
        action_one_hot = np.zeros(ACTION_DIM)
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

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")


# Main script
if __name__ == "__main__":
    # Inizializza l'ambiente
    env = gym.make("Acrobot-v1")

    # Raccogli i dati
    print("Raccolta dei dati...")
    transitions = collect_data(env, ENV_STEPS)
    dataset = TransitionDataset(transitions)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Inizializza il modello, l'ottimizzatore e la funzione di perdita
    model = TransitionModel(STATE_DIM, ACTION_DIM, hidden_units=HIDDEN_UNITS)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Addestramento
    print("Inizio addestramento...")
    train(model, dataloader, optimizer, criterion, EPOCHS)

    # Salvataggio del modello
    torch.save(model.state_dict(), "transition_model.pth")
    print("Modello salvato in 'transition_model.pth'")

