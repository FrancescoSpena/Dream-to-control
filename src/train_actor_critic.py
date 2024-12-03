import torch
import torch.optim as optim
from TransitionModel import TransitionModel  
from RewardModel import RewardModel          
from PolicyModel import PolicyModel          
from ValueModel import ValueModel            

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

state_dim = 6    
action_dim = 3   
num_episodes = 2500
imagination_horizon = 15

transition_model = TransitionModel(state_dim, action_dim).to(device)
reward_model = RewardModel(state_dim).to(device)

path_transition = '/home/francesco/Desktop/Dream-to-control/models/transition_model.pth'
path_reward = '/home/francesco/Desktop/Dream-to-control/models/reward_model.pth'

transition_model.load_state_dict(torch.load(path_transition, map_location=device))
reward_model.load_state_dict(torch.load(path_reward, map_location=device))
transition_model.eval()
reward_model.eval()

# Inizializza i modelli di policy e valore sul dispositivo
policy_model = PolicyModel(input_dim=state_dim, action_dim=action_dim).to(device)
value_model = ValueModel(input_dim=state_dim).to(device)

# Ottimizzatori
policy_optimizer = optim.Adam(policy_model.parameters(), lr=6e-4)
value_optimizer = optim.Adam(value_model.parameters(), lr=8e-5)

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
    
    max_grad_norm = 10
    factor_entropy = 0.01

    for episode in range(num_episodes):
        state_latent = torch.randn((1, state_dim), dtype=torch.float32, device=device)
        state_latent = (state_latent - state_latent.mean()) / (state_latent.std() + 1e-8)


        states, actions, rewards = [], [], []
        for _ in range(imagination_horizon):
            
            action_probs = policy_model(state_latent)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            action_one_hot = torch.nn.functional.one_hot(action, num_classes=action_dim).float().to(device)

            next_state_latent = transition_model(state_latent.detach(), action_one_hot)
            reward = reward_model(state_latent.detach()).squeeze()

            states.append(state_latent)
            actions.append(action)
            rewards.append(reward)

            state_latent = next_state_latent

        states = torch.cat(states, dim=0)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)

        # Critic Update (Value Model)
        values = value_model(states)
        with torch.no_grad():
            v_lambda = compute_v_lambda(rewards, values)
        value_loss = ((values.squeeze() - v_lambda) ** 2).mean()

        value_optimizer.zero_grad()
        value_loss.backward()  
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_grad_norm)
        value_optimizer.step()

        # Detach states before policy update
        states = states.detach()
        actions = actions.detach()

        # Actor Update (Policy Model)
        action_probs = policy_model(states)  
        action_distribution = torch.distributions.Categorical(action_probs)
        log_probs = action_distribution.log_prob(actions)
        with torch.no_grad():
            advantages = (v_lambda - values.detach()).squeeze()
        
        entropy = action_distribution.entropy().mean()
        policy_loss = -(log_probs * advantages).mean() - factor_entropy * entropy

        policy_optimizer.zero_grad()
        policy_loss.backward()  
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
        policy_optimizer.step()

    
        if episode % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Value Loss: {value_loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}")



# Salvataggio dei modelli
def save_models(policy_model, value_model, policy_path="models/policy_model.pth", value_path="models/value_model.pth"):
    torch.save(policy_model.state_dict(), policy_path)
    torch.save(value_model.state_dict(), value_path)
    print(f"Modelli salvati in:\n - Policy Model: {policy_path}\n - Value Model: {value_path}")

# Main Script
if __name__ == "__main__":
    print("Inizio addestramento dei modelli...")
    train_models(policy_model, value_model, num_episodes=num_episodes, imagination_horizon=imagination_horizon)
    save_models(policy_model, value_model)
