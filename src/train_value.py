import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from ValueModel import ValueModel
from Dataset import ValueDataset

STATE_DIM = 6  
ACTION_DIM = 3
HIDDEN_UNITS = [128, 128]  
LEARNING_RATE = 1e-3  
BATCH_SIZE = 64  
EPOCHS = 50  
DISCOUNT_FACTOR = 0.99  
EPSILON = 0.1  
NUM_EPISODES = 100 


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_data(env, num_episodes, value_model, epsilon=0.1, discount_factor=0.99, device="cpu"):
    states = []
    target_values = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_states = []
        rewards = []

        done = False
        while not done:
            # Converti lo stato in un tensor
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            # Epsilon-greedy: scegli l'azione
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Esplorazione casuale
            else:
                # Sfrutta il Value Model per scegliere l'azione che minimizza il valore (reward negativa)
                with torch.no_grad():
                    # Calcola i valori delle azioni previste
                    q_values = [
                        value_model(torch.cat([state_tensor, torch.eye(env.action_space.n, device=device)[a].unsqueeze(0)], dim=-1)).item()
                        for a in range(env.action_space.n)
                    ]
                action = np.argmin(q_values)  # Azione con il minimo valore previsto

            # Interagisci con l'ambiente
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Salva lo stato e la ricompensa
            episode_states.append(state)
            rewards.append(reward)

            state = next_state
            done = terminated or truncated

        # Calcola i valori cumulativi scontati per l'episodio
        returns = compute_discounted_rewards(rewards, discount_factor)
        states.extend(episode_states)
        target_values.extend(returns)

    return states, target_values

def compute_discounted_rewards(rewards, discount_factor):
    discounted_rewards = []
    cumulative_reward = 0.0
    for reward in reversed(rewards):
        cumulative_reward = reward + discount_factor * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)  # Inserisci in ordine inverso
    return discounted_rewards


def train_value_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for state, target_value in dataloader:
            optimizer.zero_grad()

            # Predizione del valore
            predicted_value = model(state)

            # Calcolo della perdita
            loss = criterion(predicted_value, target_value)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")


if __name__ == '__main__':
    env = gym.make("Acrobot-v1")
    value_model = ValueModel(input_dim=STATE_DIM + ACTION_DIM).to(device)

    # Raccogli i dati con epsilon-greedy
    print("Raccolta dei dati...")
    states, target_values = collect_data(
        env, num_episodes=NUM_EPISODES, value_model=value_model, epsilon=EPSILON, discount_factor=DISCOUNT_FACTOR, device=device
    )
    print(f"Numero di stati raccolti: {len(states)}")

    # Crea il dataset e il dataloader
    dataset = ValueDataset(states, target_values)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Configura l'ottimizzatore e la funzione di perdita
    optimizer = torch.optim.Adam(value_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Addestra il Value Model
    print("Inizio addestramento del Value Model...")
    train_value_model(value_model, dataloader, optimizer, criterion, EPOCHS)

    # Salva il modello addestrato
    torch.save(value_model.state_dict(), "value_model.pth")
    print("Modello salvato in 'value_model.pth'")