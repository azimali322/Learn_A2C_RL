# Import necessary libraries
import os  # For directory operations
import gymnasium as gym  # For the CartPole environment
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting training results
from stable_baselines3 import A2C  # The A2C algorithm implementation
from stable_baselines3.common.vec_env import DummyVecEnv  # For vectorized environments
from stable_baselines3.common.callbacks import EvalCallback  # For evaluation during training
from stable_baselines3.common.monitor import Monitor  # For monitoring training progress
from tqdm import tqdm  # For progress bars

def create_env(env_name, log_dir, is_eval=False):
    """Create and wrap an environment with monitoring."""
    env = gym.make(env_name)
    env = Monitor(env, log_dir + ("eval/" if is_eval else ""))
    return DummyVecEnv([lambda: env])

def train_agent(env_name, total_timesteps=50000):
    """Train the A2C agent on the specified environment."""
    # Create directories for logs and models
    log_dir = f"logs/{env_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir + "models/", exist_ok=True)
    
    # Set up environments
    env = create_env(env_name, log_dir)
    eval_env = create_env(env_name, log_dir, is_eval=True)
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "models/best_model",
        log_path=log_dir + "eval_logs/",
        eval_freq=1000,
        deterministic=True,
        render=True
    )
    
    # Initialize and train the agent
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        n_steps=5,
        gamma=0.99,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    try:
        print(f"Starting training on {env_name}... Press Ctrl+C to stop and save the best model.")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. The best model has been saved.")
    
    # Save the final model
    model.save(log_dir + "models/final_model")
    return model, log_dir

def test_agent(model, env_name, num_episodes=5, render=True):
    """Test the trained agent on multiple episodes."""
    env = create_env(env_name, "logs/test/")
    results = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            episode_length += 1
            if render:
                env.envs[0].render()
        
        results.append({
            'episode': episode + 1,
            'total_reward': total_reward,
            'episode_length': episode_length
        })
        print(f"Episode {episode + 1} - Total reward: {total_reward}, Length: {episode_length}")
    
    return pd.DataFrame(results)

def plot_training_results(log_dir):
    """Plot and save training metrics."""
    # Read the training logs
    data = pd.read_csv(log_dir + "monitor.csv", skiprows=1)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards
    ax1.plot(data['r'], label='Reward')
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot episode lengths
    ax2.plot(data['l'], label='Length', color='orange')
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(log_dir + 'training_metrics.png')
    plt.close()
    
    # Also save the raw data
    data.to_csv(log_dir + 'training_metrics.csv', index=False)

def main():
    # Train on CartPole-v1
    env_name = "CartPole-v1"
    model, log_dir = train_agent(env_name)
    
    # Plot training results
    plot_training_results(log_dir)
    
    # Test the trained agent
    print("\nTesting the trained agent...")
    test_results = test_agent(model, env_name, num_episodes=5, render=True)
    
    # Save test results
    test_results.to_csv(log_dir + "test_results.csv", index=False)
    print("\nTest results saved to", log_dir + "test_results.csv")

if __name__ == "__main__":
    main() 