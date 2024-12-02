import gymnasium as gym
import torch
import numpy as np

from TransitionModel import TransitionModel
from RewardModel import RewardModel
from PolicyModel import PolicyModel
from ValueModel import ValueModel

# Funzione per selezionare azioni
def select_action(state, policy_model, device="cpu"):
    state = state.to(device)
    with torch.no_grad():
        action_probs = policy_model(state)  
    return action_probs.argmax(dim=-1)

# Ciclo principale
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    env = gym.make("Acrobot-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    transition_model = TransitionModel(state_dim, action_dim).to(device)
    reward_model = RewardModel(state_dim).to(device)
    policy_model = PolicyModel(state_dim, action_dim).to(device)
    value_model = ValueModel(state_dim).to(device)

    transition_model.load_state_dict(torch.load("transition_model.pth", map_location=device))
    reward_model.load_state_dict(torch.load("reward_model.pth", map_location=device))
    policy_model.load_state_dict(torch.load("policy_model.pth", map_location=device))
    value_model.load_state_dict(torch.load("value_model.pth", map_location=device))

    state, _ = env.reset()
    done = False
    step = 0
    rewards = []
    episode = 1

    for ep in range(episode): 
        total_reward = 0
        while not done: 
            env.render()  
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action_tensor = select_action(state_tensor, policy_model, device)
            state, reward, done, _, _ = env.step(action_tensor.item())
            
            #step +=1 
        
    print('Mean Reward:', np.mean(reward))

    env.close()
