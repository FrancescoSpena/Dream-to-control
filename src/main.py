import torch
import gymnasium as gym
from PolicyModel import PolicyModel
import argparse

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
    while not (terminated or truncated):
        with torch.no_grad():
            action_probs = policy_model(state_tensor)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

    return total_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    num_episodes = 2

    if args.render:
        env = gym.make('Acrobot-v1', render_mode='human')
    else:
        env = gym.make('Acrobot-v1')

    print("Simulating policy...")
    rewards = []
    for episode in range(num_episodes):
        total_reward = simulate_episode(env, policy_model)
        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Total Reward = {total_reward:.2f}")

    env.close()
    print(f"Average Reward over {num_episodes} episodes: {sum(rewards) / num_episodes:.2f}")