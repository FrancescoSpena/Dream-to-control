import gymnasium as gym
from stable_baselines3 import PPO
import torch
import matplotlib.pyplot as plt
import numpy as np


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = gym.make("Acrobot-v1")
model = PPO("MlpPolicy", env, device=device)

# Load the saved weights into the policy
model.policy.load_state_dict(torch.load("ppo_acrobot.pth", map_location=device))
print("Model loaded from ppo_acrobot.pth")


if __name__ == '__main__':

    value_reward = []
    episodes = 10000
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0

        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)

            next_state, reward, terminated, truncated, info = env.step(action)
            obs = next_state
            total_reward += reward
            done = terminated or truncated
        
        
        value_reward.append(total_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    
    plot_loss_with_average(value_reward)
    env.close()

    
