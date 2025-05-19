import numpy as np
import matplotlib.pyplot as plt
from environment import MountainCarEnv
from a2c_agent import A2CAgent
import torch
from collections import deque
import time
import os
import sklearn.preprocessing

def plot_learning_curve(scores, avg_scores, max_positions, filename):
    plt.figure(figsize=(12, 8))
    
    # Plot scores
    plt.subplot(2, 1, 1)
    plt.plot(scores, alpha=0.2, color='blue', label='Episode Scores')
    plt.plot(avg_scores, color='red', label='Moving Average')
    plt.axhline(y=90, color='g', linestyle='--', label='Solving Criterion')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Learning Curve - Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot maximum positions
    plt.subplot(2, 1, 2)
    plt.plot(max_positions, alpha=0.6, color='green', label='Max Position')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Goal Position')
    plt.xlabel('Episode')
    plt.ylabel('Maximum Position')
    plt.title('Learning Curve - Maximum Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train(max_episodes=5000, eval_every=100):
    # Create save directories
    save_dir = 'saved_models'
    plot_dir = 'training_plots'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = MountainCarEnv()
    
    # Initialize state scaler
    state_space_samples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)
    
    # Initialize agent with notebook architecture
    agent = A2CAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=400,  # Increased from 256 to match notebook
        lr_actor=0.00002,  # Separate learning rates for actor and critic
        lr_critic=0.001,
        gamma=0.99,
        state_scaler=scaler  # Pass the scaler to the agent
    )
    
    # Training metrics
    scores = []
    max_positions = []
    best_reward = float('-inf')
    best_position = float('-inf')
    
    # Success tracking
    success_window = deque(maxlen=100)
    episodes_since_improvement = 0
    
    try:
        for episode in range(max_episodes):
            state = env.reset()
            episode_score = 0
            episode_max_position = -np.inf
            done = False
            
            while not done:
                # Get action and step
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Track maximum position
                episode_max_position = max(episode_max_position, next_state[0])
                
                # Store transition and learn
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_score += reward
                
                # Learn from experience
                if len(agent.replay_buffer.buffer) >= agent.min_replay_size:
                    agent.learn()
            
            # Update metrics
            scores.append(episode_score)
            max_positions.append(episode_max_position)
            success_window.append(1 if episode_max_position >= 0.5 else 0)
            
            # Save best model
            if episode_score > best_reward:
                best_reward = episode_score
                agent.save_model(os.path.join(save_dir, 'best_reward_model.pth'))
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 1
            
            if episode_max_position > best_position:
                best_position = episode_max_position
                agent.save_model(os.path.join(save_dir, 'best_position_model.pth'))
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_score = np.mean(scores[-10:])
                avg_position = np.mean(max_positions[-10:])
                success_rate = np.mean(success_window) * 100
                print(f'Episode {episode + 1}/{max_episodes}')
                print(f'Average Score: {avg_score:.2f}')
                print(f'Average Position: {avg_position:.3f}')
                print(f'Success Rate: {success_rate:.1f}%')
                print(f'Best Reward: {best_reward:.2f}')
                print(f'Best Position: {best_position:.3f}')
                print(f'Episodes Since Improvement: {episodes_since_improvement}')
                print('-' * 50)
            
            # Plot learning curves
            if (episode + 1) % eval_every == 0:
                plot_learning_curve(
                    scores,
                    np.convolve(scores, np.ones(100)/100, mode='valid'),
                    max_positions,
                    os.path.join(plot_dir, f'learning_curve_episode_{episode+1}.png')
                )
            
            # Check early stopping
            if episodes_since_improvement >= 1000:
                print('\nStopping early due to lack of improvement')
                break
            
            # Check solving criteria (matching notebook's criteria)
            if len(success_window) >= 100:
                success_rate = np.mean(success_window)
                if np.mean(scores[-100:]) > 90:  # Changed to match notebook's criterion
                    print(f'\nEnvironment solved in {episode + 1} episodes!')
                    print(f'Mean cumulative reward over 100 episodes: {np.mean(scores[-100:]):.2f}')
                    break
        
        # Save final model and plots
        agent.save_model(os.path.join(save_dir, 'final_model.pth'))
        plot_learning_curve(
            scores,
            np.convolve(scores, np.ones(100)/100, mode='valid'),
            max_positions,
            os.path.join(plot_dir, 'final_learning_curve.png')
        )
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save_model(os.path.join(save_dir, 'interrupted_model.pth'))
    
    finally:
        env.close()
        return scores, max_positions

if __name__ == "__main__":
    # Train the agent
    scores, max_positions = train(max_episodes=5000, eval_every=100)
    
    # Plot final results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(max_positions)
    plt.title('Maximum Positions')
    plt.xlabel('Episode')
    plt.ylabel('Position')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Goal Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 