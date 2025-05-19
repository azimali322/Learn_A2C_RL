import numpy as np
import matplotlib.pyplot as plt
from environment import MountainCarEnv
from a2c_agent import A2CAgent
import torch
from collections import deque
from tqdm import tqdm
import time
import os

def plot_curriculum_progress(curriculum_levels, success_rates, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(curriculum_levels, color='blue', label='Curriculum Level')
    plt.plot(success_rates, color='green', label='Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Curriculum Learning Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def train_curriculum(max_episodes=10000, eval_every=100):
    # Create save directories
    save_dir = 'saved_models'
    plot_dir = 'training_plots'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = MountainCarEnv()
    agent = A2CAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=256,
        lr=0.001,
        gamma=0.99
    )
    
    # Training metrics
    scores = []
    curriculum_levels = []
    success_rates = []
    level_completion_times = {i: [] for i in range(11)}  # Track time spent at each level
    
    # Success tracking per level
    level_success_window = deque(maxlen=50)
    current_level_episodes = 0
    
    # Best model tracking
    best_success_rate = 0.0
    episodes_since_improvement = 0
    
    try:
        pbar = tqdm(range(max_episodes), desc='Training', unit='episode')
        for episode in pbar:
            state = env.reset()
            episode_score = 0
            episode_steps = 0
            max_position = -np.inf
            done = False
            
            # Store initial curriculum level
            start_level = env.curriculum_level
            
            while not done:
                # Get action and step
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Track maximum position
                max_position = max(max_position, next_state[0])
                
                # Store transition and learn
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_score += reward
                episode_steps += 1
            
            # Update metrics
            scores.append(episode_score)
            curriculum_levels.append(env.curriculum_level)
            level_completion_times[start_level].append(episode_steps)
            
            # Track success for current level
            success = max_position >= 0.5
            level_success_window.append(1 if success else 0)
            current_level_episodes += 1
            
            # Calculate success rate for current level
            if len(level_success_window) >= 20:  # Minimum episodes before considering level change
                success_rate = sum(level_success_window) / len(level_success_window)
                success_rates.append(success_rate)
                
                # Check for level advancement
                if success_rate >= 0.2 and current_level_episodes >= 50:  # Modified threshold and minimum episodes
                    if env.curriculum_level < env.max_curriculum_level:
                        env.curriculum_level += 1
                        print(f"\nAdvancing to curriculum level {env.curriculum_level}")
                        print(f"Success rate at level {env.curriculum_level-1}: {success_rate:.2f}")
                        print(f"Average steps to complete: {np.mean(level_completion_times[env.curriculum_level-1]):.1f}")
                        
                        # Reset tracking for new level
                        level_success_window.clear()
                        current_level_episodes = 0
                
                # Save best model
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    agent.save_model('best_curriculum_model.pth')
                    episodes_since_improvement = 0
                    print(f"\nNew best model saved! Success rate: {success_rate:.3f}")
                else:
                    episodes_since_improvement += 1
            
            # Periodic evaluation and plotting
            if episode % eval_every == 0:
                # Plot progress
                plot_curriculum_progress(
                    curriculum_levels,
                    success_rates,
                    os.path.join(plot_dir, 'curriculum_progress.png')
                )
                
                # Print progress
                print(f"\nEpisode {episode}/{max_episodes}")
                print(f"Current Curriculum Level: {env.curriculum_level}")
                print(f"Success Rate: {np.mean(level_success_window)*100:.1f}%")
                print(f"Episodes at Current Level: {current_level_episodes}")
                print(f"Episodes Since Improvement: {episodes_since_improvement}")
            
            # Update progress bar
            pbar.set_postfix({
                'Level': env.curriculum_level,
                'Score': f'{episode_score:.1f}',
                'MaxPos': f'{max_position:.2f}'
            })
            
            # Early stopping if stuck at a level too long
            if episodes_since_improvement >= 1000:
                print("\nStopping early due to lack of improvement")
                break
        
        # Final evaluation
        print("\nTraining completed!")
        print(f"Final Curriculum Level: {env.curriculum_level}")
        print(f"Best Success Rate: {best_success_rate:.3f}")
        
        # Save final plots
        plot_curriculum_progress(
            curriculum_levels,
            success_rates,
            os.path.join(plot_dir, 'final_curriculum_progress.png')
        )
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        env.close()

if __name__ == '__main__':
    train_curriculum() 