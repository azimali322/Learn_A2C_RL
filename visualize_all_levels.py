import numpy as np
import matplotlib.pyplot as plt
from environment import MountainCarEnv
from a2c_agent import A2CAgent
import torch
import time
from collections import defaultdict

def visualize_episode(env, agent, episode_data=None):
    """Visualize a single episode and collect data"""
    if episode_data is None:
        episode_data = defaultdict(list)
    
    state = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    
    while not done:
        # Store state information
        episode_data['positions'].append(state[0])
        episode_data['velocities'].append(state[1])
        
        # Get action probabilities
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(agent.device)
            log_probs, value = agent.network(state_tensor)
            probs = agent.get_action_probs(log_probs)
            action = torch.argmax(probs).item()
        
        # Store action probabilities and value
        episode_data['action_probs'].append(probs[0].cpu().numpy())
        episode_data['values'].append(value.item())
        
        # Take action
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        steps += 1
        
        # Store action and reward
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        
        state = next_state
        env.render()
        time.sleep(0.05)  # Slow down visualization
    
    episode_data['total_reward'] = episode_reward
    episode_data['steps'] = steps
    episode_data['final_position'] = state[0]
    return episode_data

def main():
    # Initialize agent
    agent = A2CAgent(
        state_dim=2,  # position and velocity
        n_actions=3,  # left, nothing, right
        hidden_dim=256
    )
    
    # Load the best model
    agent.load_model('best_overall_model.pth')
    
    # Test on different curriculum levels
    curriculum_levels = [0, 2, 4, 6, 8, 10]  # Test a range of difficulties
    
    print("\nTesting agent across different curriculum levels:")
    print("=" * 50)
    
    for level in curriculum_levels:
        # Initialize environment with specific curriculum level
        env = MountainCarEnv(render_mode='human', curriculum_level=level)
        
        print(f"\nCurriculum Level {level}")
        print("-" * 20)
        
        # Run one episode
        episode_data = visualize_episode(env, agent)
        
        # Print episode statistics
        print(f"Starting Position: {episode_data['positions'][0]:.3f}")
        print(f"Starting Velocity: {episode_data['velocities'][0]:.3f}")
        print(f"Steps to Complete: {episode_data['steps']}")
        print(f"Final Position: {episode_data['final_position']:.3f}")
        print(f"Total Reward: {episode_data['total_reward']:.2f}")
        
        # Close environment
        env.close()
        
        # Pause between episodes
        time.sleep(1)

if __name__ == '__main__':
    main() 