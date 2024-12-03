import gymnasium as gym 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import Counter

from TransitionModel import TransitionModel
from Dataset import TransitionDataset

def collect_data(env, steps, state_dim, action_dim):
    transitions = []
    state, _ = env.reset()  
    for _ in range(steps):
        action = env.action_space.sample()  
        next_state, _, terminated, truncated, done = env.step(action)

        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)

        if state.shape != (state_dim,) or next_state.shape != (state_dim,):
            continue  

        action_one_hot = np.zeros(action_dim)
        action_one_hot[action] = 1

        transitions.append((state, action_one_hot, next_state))
        state = next_state

        if done:
            state, _ = env.reset()  
    return transitions

def collect_balanced_data(env, steps_per_action, state_dim, action_dim):
    transitions = {action: [] for action in range(action_dim)}
    total_steps = steps_per_action * action_dim
    state, _ = env.reset()

    while sum(len(v) for v in transitions.values()) < total_steps:
        for action in range(action_dim):
            if len(transitions[action]) < steps_per_action:
                next_state, _, terminated, truncated, _ = env.step(action)

                # Ensure state and next_state are numpy arrays of the correct shape
                if not isinstance(state, np.ndarray):
                    state = np.array(state)
                if not isinstance(next_state, np.ndarray):
                    next_state = np.array(next_state)

                if state.shape == (state_dim,) and next_state.shape == (state_dim,):
                    action_one_hot = np.zeros(action_dim)
                    action_one_hot[action] = 1
                    transitions[action].append((state, action_one_hot, next_state))

                state = next_state

                # Reset environment if done
                if terminated or truncated:
                    state, _ = env.reset()

    # Flatten the transitions into a single list
    balanced_transitions = [
        transition for action_transitions in transitions.values() for transition in action_transitions
    ]

    return balanced_transitions


def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    loss_value = []  
    for epoch in range(epochs):
        epoch_loss = 0.0
        for state, action, next_state in dataloader:
            optimizer.zero_grad()

            predicted_next_state = model(state, action)

            loss = criterion(predicted_next_state, next_state)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_value.append(avg_loss)  

        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return loss_value

def check_balance(transitions, action_dim):
    action_counts = Counter()

    for _, action_one_hot, _ in transitions:
        action = np.argmax(action_one_hot)  
        action_counts[action] += 1

    for action in range(action_dim):
        print(f"Action {action}: {action_counts[action]} samples")
    
    return action_counts


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



if __name__ == "__main__":
    env = gym.make("Acrobot-v1")

    bath_size = 32
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 

    epochs = 200
    lr = 1e-4
    num_ep = 10

    print("Collect data...")
    transitions = collect_balanced_data(env,steps_per_action=100,state_dim=state_dim, action_dim=action_dim)
    check_balance(transitions,action_dim=action_dim)

    dataset = TransitionDataset(transitions)
    dataloader = DataLoader(dataset, batch_size=bath_size, shuffle=True)

    model = TransitionModel(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Starting training...")
    loss_value = train(model, dataloader, optimizer, criterion, epochs)
    plot_loss_with_average(loss_value,epochs)

    torch.save(model.state_dict(), "models/transition_model.pth")
    print("Model saved as 'transition_model.pth' in models folder")

