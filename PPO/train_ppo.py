import gymnasium as gym
from stable_baselines3 import PPO
import torch

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create and train the model on the GPU
env = gym.make("Acrobot-v1")
model = PPO("MlpPolicy", env, verbose=1, device=device)

# Train the model
model.learn(total_timesteps=200000)

# Save the model parameters in a .pth file
torch.save(model.policy.state_dict(), "ppo_acrobot.pth")
print("Model saved as ppo_acrobot.pth")
