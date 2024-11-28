import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from Policy import PolicyModel
from ValueModel import ValueModel


def compute_advantages(rewards, values, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = returns - torch.tensor(values, dtype=torch.float32)
    return advantages


def train_policy_model(env, policy_model, value_model, optimizer_policy, episodes=500, gamma=0.99, device='cpu'):
    value_model.eval()  # Assicurati che il Value Model sia in modalità valutazione

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)

        log_probs = []
        rewards = []
        values = []

        done = False
        while not done:
            # Ottieni le probabilità delle azioni dalla politica
            action_probs = policy_model(state)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()

            # Calcola il valore stimato
            with torch.no_grad():
                value = value_model(state)

            # Esegui l'azione
            next_state, reward, done, _, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

            # Memorizza log-prob, reward e valore
            log_probs.append(action_distribution.log_prob(action))
            rewards.append(reward)
            values.append(value.item())

            # Passa allo stato successivo
            state = next_state

        # Calcola vantaggi e loss della politica
        advantages = compute_advantages(rewards, values, gamma).to(device)
        policy_loss = -torch.sum(torch.stack(log_probs) * advantages.detach())

        # Aggiorna il modello di policy
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Stampa i risultati
        print(f"Episode {episode + 1}/{episodes}, Reward: {sum(rewards):.2f}, Policy Loss: {policy_loss.item():.4f}")


def main():
    # Configurazioni
    env_name = "Acrobot-v1"
    episodes = 500
    gamma = 0.99
    learning_rate_policy = 1e-3
    hidden_units_policy = [128, 128]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Inizializza l'ambiente
    env = gym.make(env_name)

    # Carica il Value Model pre-addestrato
    state_dim = env.observation_space.shape[0]
    value_model = ValueModel(input_dim=state_dim)
    value_model.load_state_dict(torch.load("value_model.pth"))
    value_model.to(device)

    # Inizializza il Policy Model
    action_dim = env.action_space.n
    policy_model = PolicyModel(input_dim=state_dim, action_dim=action_dim, hidden_units=hidden_units_policy).to(device)

    # Inizializza l'ottimizzatore per la politica
    optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate_policy)

    # Addestramento del Policy Model
    print("Training policy model...")
    train_policy_model(env, policy_model, value_model, optimizer_policy, episodes=episodes, gamma=gamma, device=device)

    # Salva il Policy Model
    torch.save(policy_model.state_dict(), "policy_model.pth")
    print("Training complete. Policy model saved.")


if __name__ == "__main__":
    main()
