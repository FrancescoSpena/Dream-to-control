import torch
import torch.optim as optim
from TransitionModel import TransitionModel  # Modello di transizione
from RewardModel import RewardModel          # Modello di reward
from PolicyModel import PolicyModel          # Modello di policy
from ValueModel import ValueModel            # Modello di valore

# Configurazione del dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configurazione dei parametri
state_dim = 6    # Dimensione dello stato per Acrobot-v1
action_dim = 3   # Numero di azioni discrete
num_episodes = 100  # Numero di episodi di training
imagination_horizon = 15  # Lunghezza delle traiettorie immaginate

# Carica i modelli pre-addestrati e spostali sul dispositivo
transition_model = TransitionModel(state_dim, action_dim).to(device)
reward_model = RewardModel(state_dim).to(device)
transition_model.load_state_dict(torch.load("transition_model.pth", map_location=device))
reward_model.load_state_dict(torch.load("reward_model.pth", map_location=device))
transition_model.eval()
reward_model.eval()

# Inizializza i modelli di policy e valore sul dispositivo
policy_model = PolicyModel(input_dim=state_dim, action_dim=action_dim).to(device)
value_model = ValueModel(input_dim=state_dim).to(device)

# Ottimizzatori
policy_optimizer = optim.Adam(policy_model.parameters(), lr=1e-4)
value_optimizer = optim.Adam(value_model.parameters(), lr=1e-4)

# Funzione per calcolare V_lambda
def compute_v_lambda(rewards, values, gamma=0.99, lambda_=0.95):
    H = len(rewards)
    v_lambda = torch.zeros(H, device=device)
    gae = 0.0
    next_value = values[-1]
    for t in reversed(range(H)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * gae
        v_lambda[t] = gae + values[t]
        next_value = values[t]
    return v_lambda

# Training Loop
def train_models(policy_model, value_model, num_episodes=100, imagination_horizon=15):
    for episode in range(num_episodes):
        # Stato iniziale casuale nello spazio latente
        state_latent = torch.randn((1, state_dim), dtype=torch.float32, device=device)

        # Immaginazione
        states, actions, rewards = [], [], []
        for _ in range(imagination_horizon):
            # Politica attuale
            action_probs = policy_model(state_latent)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            action_one_hot = torch.nn.functional.one_hot(action, num_classes=action_dim).float().to(device)

            # Modello di transizione e reward
            next_state_latent = transition_model(state_latent, action_one_hot)
            reward = reward_model(state_latent).squeeze()

            # Salva le transizioni
            states.append(state_latent)
            actions.append(action)
            rewards.append(reward)

            # Aggiorna lo stato latente
            state_latent = next_state_latent

        states = torch.cat(states, dim=0)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, device=device)

        # Critic Update (Value Model)
        values = value_model(states)
        with torch.no_grad():
            v_lambda = compute_v_lambda(rewards, values)
        value_loss = ((values.squeeze() - v_lambda) ** 2).mean()
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Actor Update (Policy Model)
        with torch.no_grad():
            advantages = (v_lambda - values).squeeze()
        action_probs = policy_model(states)
        action_distribution = torch.distributions.Categorical(action_probs)
        log_probs = action_distribution.log_prob(actions)
        policy_loss = -(log_probs * advantages).mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}, Value Loss: {value_loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}")

# Salvataggio dei modelli
def save_models(policy_model, value_model, policy_path="policy_model.pth", value_path="value_model.pth"):
    torch.save(policy_model.state_dict(), policy_path)
    torch.save(value_model.state_dict(), value_path)
    print(f"Modelli salvati in:\n - Policy Model: {policy_path}\n - Value Model: {value_path}")

# Main Script
if __name__ == "__main__":
    print("Inizio addestramento dei modelli...")
    train_models(policy_model, value_model, num_episodes=num_episodes, imagination_horizon=imagination_horizon)
    save_models(policy_model, value_model)
