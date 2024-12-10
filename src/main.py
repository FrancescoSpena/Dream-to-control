import torch
import gymnasium as gym
from PolicyModel import PolicyModel
import argparse
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

state_dim = 6  
action_dim = 3  
policy_path = "../models/Original_loss_policy_model.pth"

policy_model = PolicyModel(input_dim=state_dim, action_dim=action_dim).to(device)
policy_model.load_state_dict(torch.load(policy_path, map_location=device))
policy_model.eval()


def simulate_episode(env, policy_model):
    state, _ = env.reset()  
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0

    terminated = False
    truncated = False
    done = False
    while not (terminated or truncated):
        with torch.no_grad():
            action_probs = policy_model(state_tensor)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample().item()

        next_state, reward, terminated, truncated, done = env.step(action)
        total_reward += reward

        state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

    return total_reward

def plot_loss_with_average(loss_values):
    if not loss_values:
        raise ValueError("loss_values is empty.")
    
    epochs = len(loss_values)
    epoch_numbers = list(range(1, epochs + 1))
    cumulative_average = np.cumsum(loss_values) / np.arange(1, epochs + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, loss_values, label='Loss', color='blue', alpha=0.6)
    plt.plot(epoch_numbers, cumulative_average, label='Average Loss', color='orange', linewidth=2)
    
    plt.title('Reward Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_reward_comparison_with_average(reward_original, reward_ppo):
    if not reward_original:
        raise ValueError("reward_original is empty.")
    if not reward_ppo:
        raise ValueError("reward_ppo is empty.")
    if len(reward_original) != len(reward_ppo):
        raise ValueError("reward_original and reward_ppo must have the same length.")
    
    epochs = len(reward_original)
    epoch_numbers = list(range(1, epochs + 1))
    
    # Calcolo delle medie cumulative
    cumulative_average_original = np.cumsum(reward_original) / np.arange(1, epochs + 1)
    cumulative_average_ppo = np.cumsum(reward_ppo) / np.arange(1, epochs + 1)
    
    # Creazione del grafico
    plt.figure(figsize=(12, 7))
    
    # Plot del reward originale e della sua media cumulativa
    plt.plot(epoch_numbers, reward_original, label='Reward Original Model', color='#4DBEEE', alpha=0.6)  # Blu chiaro
    plt.plot(epoch_numbers, cumulative_average_original, label='Average Reward Original Model', color='#0072BD', linewidth=2)  # Blu scuro
    
    # Plot del reward PPO e della sua media cumulativa 
    plt.plot(epoch_numbers, reward_ppo, label='Reward PPO', color='#d0e9af', alpha=0.6)  
    plt.plot(epoch_numbers, cumulative_average_ppo, label='Average Reward PPO', color='#77AC30', linewidth=2) 
    # Titoli e etichette degli assi
    plt.title('Comparison of Original and PPO Rewards')
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Mostra il grafico
    plt.show()

def save_rewards_to_csv(rewards, reward_ppo, filename):
    data = {
        "Original_Reward": rewards,
        "PPO_Reward": reward_ppo
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Rewards saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--ep', action='store_true')
    args = parser.parse_args()

    num_episodes = 10000

    if args.render:
        env = gym.make('Acrobot-v1', render_mode='human')
    else:
        env = gym.make('Acrobot-v1')

    #PPO Model 
    ppo_model = PPO("MlpPolicy", env, device=device)
    ppo_model.policy.load_state_dict(torch.load("../PPO/ppo_acrobot.pth", map_location=device))

    print("Simulating policy...")
    rewards = []
    reward_ppo = []
    for episode in range(num_episodes):
        total_reward = simulate_episode(env, policy_model)
        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Total Reward Original = {total_reward:.2f}")
    
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = ppo_model.predict(obs, deterministic=True)

            next_state, reward, terminated, truncated, info = env.step(action)
            obs = next_state
            total_reward += reward
            done = terminated or truncated
        
        reward_ppo.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Total Reward PPO = {total_reward}")

    
    plot_reward_comparison_with_average(rewards,reward_ppo)
    #plot_loss_with_average(rewards)
    save_rewards_to_csv(rewards, reward_ppo, "rewards_data.csv")
    
    env.close()
    #print(f"Average Reward over {num_episodes} episodes: {sum(rewards) / num_episodes:.2f}")
