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
        episode_data['action_probs'].append(probs[0].cpu().numpy())  # Get first batch item
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
    return episode_data

def plot_episode_data(episode_data, episode_num):
    """Plot detailed metrics from the episode"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Episode {episode_num} Analysis\nTotal Reward: {episode_data["total_reward"]:.2f}, Steps: {episode_data["steps"]}')
    
    # Position and Velocity
    ax = axes[0, 0]
    ax.plot(episode_data['positions'], label='Position')
    ax.axhline(y=0.5, color='r', linestyle='--', label='Goal Position')
    ax.set_ylabel('Position')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True)
    
    ax = axes[0, 1]
    ax.plot(episode_data['velocities'], label='Velocity')
    ax.set_ylabel('Velocity')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True)
    
    # Action Probabilities
    ax = axes[1, 0]
    action_probs = np.array(episode_data['action_probs'])
    for i in range(3):
        ax.plot(action_probs[:, i], label=f'Action {i}')
    ax.set_ylabel('Action Probability')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True)
    
    # Actions Taken
    ax = axes[1, 1]
    ax.plot(episode_data['actions'], label='Actions', drawstyle='steps-post')
    ax.set_ylabel('Action Taken')
    ax.set_xlabel('Step')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Left', 'Nothing', 'Right'])
    ax.legend()
    ax.grid(True)
    
    # Value Estimates
    ax = axes[2, 0]
    ax.plot(episode_data['values'], label='Value Estimate')
    ax.set_ylabel('Value')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True)
    
    # Rewards
    ax = axes[2, 1]
    ax.plot(episode_data['rewards'], label='Reward')
    ax.set_ylabel('Reward')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'episode_{episode_num}_analysis.png')
    plt.close()

def main():
    # Initialize environment and agent
    env = MountainCarEnv(render_mode='human')
    agent = A2CAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=256
    )
    
    # Load the best model
    agent.load_model('best_overall_model.pth')
    
    # Run and visualize episodes
    n_episodes = 3
    for episode in range(n_episodes):
        print(f"\nRunning episode {episode + 1}/{n_episodes}")
        episode_data = visualize_episode(env, agent)
        plot_episode_data(episode_data, episode + 1)
        print(f"Episode {episode + 1} completed:")
        print(f"Total Reward: {episode_data['total_reward']:.2f}")
        print(f"Steps: {episode_data['steps']}")
        print(f"Final Position: {episode_data['positions'][-1]:.3f}")
        print(f"Max Velocity: {max(map(abs, episode_data['velocities'])):.3f}")
        
        time.sleep(1)  # Pause between episodes
    
    env.close()

if __name__ == '__main__':
    main() 