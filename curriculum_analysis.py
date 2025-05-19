import numpy as np
import matplotlib.pyplot as plt
from environment import MountainCarEnv
from a2c_agent import A2CAgent
import torch
from collections import defaultdict
import seaborn as sns

def analyze_curriculum_level(env, agent, level, n_episodes=50):
    """Analyze agent performance at a specific curriculum level"""
    env.curriculum_level = level
    
    metrics = {
        'success_rate': [],
        'episode_lengths': [],
        'max_positions': [],
        'avg_velocities': [],
        'rewards': [],
        'time_to_goal': []
    }
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_length = 0
        max_position = -np.inf
        velocities = []
        total_reward = 0
        reached_goal = False
        steps_to_goal = 200  # Default to max steps if goal not reached
        
        while not done:
            # Get action from agent
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(agent.device)
                log_probs, _ = agent.network(state_tensor)
                probs = agent.get_action_probs(log_probs)
                action = torch.argmax(probs).item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update metrics
            episode_length += 1
            max_position = max(max_position, next_state[0])
            velocities.append(abs(next_state[1]))
            total_reward += reward
            
            if next_state[0] >= 0.5 and not reached_goal:
                reached_goal = True
                steps_to_goal = episode_length
            
            state = next_state
        
        # Store episode metrics
        metrics['success_rate'].append(1 if max_position >= 0.5 else 0)
        metrics['episode_lengths'].append(episode_length)
        metrics['max_positions'].append(max_position)
        metrics['avg_velocities'].append(np.mean(velocities))
        metrics['rewards'].append(total_reward)
        metrics['time_to_goal'].append(steps_to_goal)
    
    return metrics

def plot_curriculum_analysis(all_metrics):
    """Plot analysis of performance across curriculum levels"""
    levels = sorted(all_metrics.keys())
    n_metrics = 6
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Curriculum Learning Analysis', fontsize=16, y=0.95)
    
    # Success Rate
    ax = axes[0, 0]
    success_rates = [np.mean(all_metrics[level]['success_rate']) * 100 for level in levels]
    ax.bar(levels, success_rates)
    ax.set_title('Success Rate by Curriculum Level')
    ax.set_xlabel('Curriculum Level')
    ax.set_ylabel('Success Rate (%)')
    ax.grid(True, alpha=0.3)
    
    # Episode Length
    ax = axes[0, 1]
    for level in levels:
        sns.kdeplot(data=all_metrics[level]['episode_lengths'], label=f'Level {level}', ax=ax)
    ax.set_title('Episode Length Distribution')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Maximum Position
    ax = axes[1, 0]
    positions = [all_metrics[level]['max_positions'] for level in levels]
    ax.boxplot(positions, labels=levels)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Goal')
    ax.set_title('Maximum Position Distribution')
    ax.set_xlabel('Curriculum Level')
    ax.set_ylabel('Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Average Velocity
    ax = axes[1, 1]
    velocities = [all_metrics[level]['avg_velocities'] for level in levels]
    ax.boxplot(velocities, labels=levels)
    ax.set_title('Average Velocity Distribution')
    ax.set_xlabel('Curriculum Level')
    ax.set_ylabel('Velocity')
    ax.grid(True, alpha=0.3)
    
    # Total Reward
    ax = axes[2, 0]
    rewards = [np.mean(all_metrics[level]['rewards']) for level in levels]
    ax.bar(levels, rewards)
    ax.set_title('Average Reward by Curriculum Level')
    ax.set_xlabel('Curriculum Level')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    
    # Time to Goal
    ax = axes[2, 1]
    for level in levels:
        times = [t for t in all_metrics[level]['time_to_goal'] if t < 200]  # Filter unsuccessful episodes
        if times:  # Only plot if there are successful episodes
            sns.kdeplot(data=times, label=f'Level {level}', ax=ax)
    ax.set_title('Time to Goal Distribution (Successful Episodes)')
    ax.set_xlabel('Steps to Reach Goal')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('curriculum_analysis.png')
    plt.close()

def print_level_summary(level, metrics):
    """Print summary statistics for a curriculum level"""
    print(f"\nCurriculum Level {level} Summary:")
    print(f"Success Rate: {np.mean(metrics['success_rate'])*100:.1f}%")
    print(f"Average Episode Length: {np.mean(metrics['episode_lengths']):.1f} steps")
    print(f"Average Max Position: {np.mean(metrics['max_positions']):.3f}")
    print(f"Average Velocity: {np.mean(metrics['avg_velocities']):.3f}")
    print(f"Average Reward: {np.mean(metrics['rewards']):.1f}")
    successful_times = [t for t in metrics['time_to_goal'] if t < 200]
    if successful_times:
        print(f"Average Time to Goal (successful episodes): {np.mean(successful_times):.1f} steps")

def main():
    # Initialize environment and agent
    env = MountainCarEnv(render_mode=None)  # No rendering for analysis
    agent = A2CAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=256
    )
    
    # Load the best model
    agent.load_model('best_overall_model.pth')
    
    # Analyze performance across curriculum levels
    all_metrics = {}
    for level in range(11):  # 0 to 10
        print(f"\nAnalyzing curriculum level {level}...")
        metrics = analyze_curriculum_level(env, agent, level)
        all_metrics[level] = metrics
        print_level_summary(level, metrics)
    
    # Plot analysis
    plot_curriculum_analysis(all_metrics)
    print("\nAnalysis complete! Check 'curriculum_analysis.png' for visualizations.")
    
    env.close()

if __name__ == '__main__':
    main() 