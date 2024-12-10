import gymnasium as gym
import matplotlib.pyplot as plt
import torch 
from stable_baselines3 import PPO

env = gym.make('Acrobot-v1', render_mode="rgb_array")
observation, info = env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = PPO("MlpPolicy", env, device=device)

# Load the saved weights into the policy
model.policy.load_state_dict(torch.load("ppo_acrobot.pth", map_location=device))
print("Model loaded from ppo_acrobot.pth")

for i in range(500):
    frame = env.render()
    plt.imsave(f"img_acrobot/acrobot_{i:04d}.png", frame)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
