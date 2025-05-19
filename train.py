import numpy as np
import matplotlib.pyplot as plt
from environment import MountainCarEnv
from a2c_agent import A2CAgent
import torch
from collections import deque
from tqdm import tqdm
import time
import os

def plot_learning_curve(scores, avg_scores, max_positions, filename, metrics=None):
    n_metrics = len(metrics) if metrics else 0
    n_subplots = 2 + n_metrics
    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4*n_subplots))
    
    # Plot scores
    axes[0].plot(scores, alpha=0.2, color='blue', label='Episode Scores')
    axes[0].plot(avg_scores, color='red', label='Moving Average')
    axes[0].axhline(y=-110, color='g', linestyle='--', label='Solving Criterion')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Learning Curve - Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot maximum positions
    axes[1].plot(max_positions, alpha=0.6, color='green', label='Max Position')
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Goal Position')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Maximum Position')
    axes[1].set_title('Learning Curve - Maximum Positions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot additional metrics if available
    if metrics:
        for i, (metric_name, metric_values) in enumerate(metrics.items(), 2):
            axes[i].plot(metric_values, alpha=0.6, label=metric_name)
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f'Training Metrics - {metric_name}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def calculate_combined_score(eval_results):
    """Calculate combined score with weighted components"""
    normalized_score = np.clip((eval_results['mean_score'] + 200) / 100, 0, 1)  # Normalize from [-200, 0] to [0, 1]
    normalized_position = eval_results['mean_position'] / 0.5  # Normalize by goal position
    success_rate = eval_results['success_rate']
    
    # Calculate position consistency (lower variance is better)
    position_std = eval_results.get('position_std', 0.5)  # Default to 0.5 if not available
    consistency = 1.0 - min(position_std, 0.5) / 0.5  # Higher is better
    
    # Weighted combination
    weights = {
        'score': 0.3,
        'position': 0.3,
        'success': 0.3,
        'consistency': 0.1
    }
    
    combined_score = (
        weights['score'] * normalized_score +
        weights['position'] * normalized_position +
        weights['success'] * success_rate +
        weights['consistency'] * consistency
    )
    
    return combined_score

def train(max_episodes=5000, eval_every=250, solving_threshold=-110.0):
    """
    Train the A2C agent with improved exploration and monitoring
    
    Args:
        max_episodes (int): Maximum number of episodes to train for
        eval_every (int): Evaluate and save metrics every N episodes
        solving_threshold (float): Score threshold for solving the environment
    """
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
    max_positions = []
    episode_lengths = []
    entropies = []
    actor_losses = []
    critic_losses = []
    
    # Moving windows for tracking
    score_window = deque(maxlen=100)
    position_window = deque(maxlen=100)
    
    # Best metrics tracking
    best_combined_score = float('-inf')
    best_success_rate = 0.0
    best_completion_time = float('inf')
    
    # Progress tracking
    start_time = time.time()
    episodes_since_improvement = 0
    max_episodes_without_improvement = 1000
    
    try:
        pbar = tqdm(range(max_episodes), desc='Training', unit='episode')
        for episode in pbar:
            state = env.reset()
            episode_score = 0
            episode_max_position = -np.inf
            episode_positions = []
            episode_entropy = 0
            steps = 0
            done = False
            
            while not done:
                # Get action and step
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Update metrics
                position = next_state[0]
                episode_positions.append(position)
                episode_max_position = max(episode_max_position, position)
                episode_score += reward
                steps += 1
                
                # Update state
                state = next_state
                
                # Learn if episode is done
                if done:
                    actor_loss, critic_loss = agent.learn()
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
            
            # Update episode metrics
            scores.append(episode_score)
            max_positions.append(episode_max_position)
            episode_lengths.append(steps)
            score_window.append(episode_score)
            position_window.append(episode_max_position)
            
            # Periodic evaluation and model saving
            if episode % eval_every == 0:
                eval_results = agent.evaluate(MountainCarEnv(), n_episodes=10)
                eval_results['position_std'] = np.std(episode_positions)
                combined_score = calculate_combined_score(eval_results)
                
                # Save models based on different criteria
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    agent.save_model('best_overall_model.pth')
                    episodes_since_improvement = 0
                    print(f'\nNew best model saved! Combined score: {combined_score:.3f}')
                
                if eval_results['success_rate'] > best_success_rate:
                    best_success_rate = eval_results['success_rate']
                    agent.save_model('best_success_rate_model.pth')
                
                # Plot learning curves
                plot_learning_curve(
                    scores,
                    np.convolve(scores, np.ones(100)/100, mode='valid'),
                    max_positions,
                    os.path.join(plot_dir, 'learning_curve.png'),
                    metrics={
                        'Actor Loss': actor_losses,
                        'Critic Loss': critic_losses,
                        'Episode Length': episode_lengths
                    }
                )
                
                # Print progress
                print(f'\nEpisode {episode}/{max_episodes}')
                print(f'Average Score: {np.mean(score_window):.2f}')
                print(f'Average Position: {np.mean(position_window):.3f}')
                print(f'Success Rate: {eval_results["success_rate"]*100:.1f}%')
                print(f'Combined Score: {combined_score:.3f}')
                print(f'Episodes Since Improvement: {episodes_since_improvement}')
            
            # Update progress bar
            if len(score_window) > 0:
                pbar.set_postfix({
                    'Avg Score': f'{np.mean(score_window):.1f}',
                    'Max Pos': f'{episode_max_position:.2f}',
                    'Steps': steps
                })
            
            # Check early stopping
            episodes_since_improvement += 1
            if episodes_since_improvement >= max_episodes_without_improvement:
                print('\nStopping early due to lack of improvement')
                break
            
            # Check solving criteria
            if len(score_window) >= 100:
                avg_score = np.mean(score_window)
                if avg_score > solving_threshold and episode_max_position >= 0.5:
                    print(f'\nEnvironment solved in {episode} episodes!')
                    print(f'Average Score: {avg_score:.2f}')
                    break
        
        # Save final model and plots
        agent.save_model('final_model.pth')
        plot_learning_curve(
            scores,
            np.convolve(scores, np.ones(100)/100, mode='valid'),
            max_positions,
            os.path.join(plot_dir, 'final_learning_curve.png'),
            metrics={
                'Actor Loss': actor_losses,
                'Critic Loss': critic_losses,
                'Episode Length': episode_lengths
            }
        )
        
        # Final evaluation
        print('\nFinal Evaluation:')
        final_eval = agent.evaluate(MountainCarEnv(), n_episodes=10)
        final_combined_score = calculate_combined_score(final_eval)
        print(f'Final Score: {final_eval["mean_score"]:.2f} ± {final_eval["std_score"]:.2f}')
        print(f'Final Position: {final_eval["mean_position"]:.3f}')
        print(f'Final Success Rate: {final_eval["success_rate"]*100:.1f}%')
        print(f'Final Combined Score: {final_combined_score:.3f}')
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save_model('interrupted_model.pth')
    finally:
        env.close()
        pbar.close()

def evaluate_saved_model(model_path, n_episodes=10, render=True):
    """Evaluate a saved model with detailed metrics"""
    env = MountainCarEnv(render_mode='human' if render else None)
    agent = A2CAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=256
    )
    
    # Load the saved model
    agent.load_model(model_path)
    
    # Evaluate
    results = agent.evaluate(env, n_episodes=n_episodes, render=render)
    combined_score = calculate_combined_score(results)
    
    print(f'\nEvaluation Results for {model_path}:')
    print(f'Mean Score: {results["mean_score"]:.2f} ± {results["std_score"]:.2f}')
    print(f'Mean Position: {results["mean_position"]:.3f}')
    print(f'Success Rate: {results["success_rate"]*100:.1f}%')
    print(f'Combined Score: {combined_score:.3f}')
    
    env.close()
    return results

if __name__ == '__main__':
    train() 