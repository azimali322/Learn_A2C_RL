# MountainCar A2C Reinforcement Learning

This project implements an Actor-Critic (A2C) reinforcement learning agent using Stable-Baselines3 to solve the MountainCar-v0 environment from Gymnasium.

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Project Structure

- `train_a2c.py`: Main training script for the A2C agent
- `requirements.txt`: List of required Python packages
- `logs/`: Directory containing training logs and saved models
  - `best_model/`: Best model saved during training
  - `eval_logs/`: Evaluation logs
  - `training_rewards.png`: Plot of training rewards
  - `episode_lengths.png`: Plot of episode lengths

## Training the Agent

To train the agent, simply run:
```bash
python train_a2c.py
```

The script will:
1. Create a MountainCar environment
2. Initialize an A2C agent
3. Train the agent for 100,000 timesteps
4. Save the best model during training
5. Test the final model
6. Generate plots of training progress

## Monitoring Training

The training progress can be monitored through:
- Console output showing training statistics
- TensorBoard logs in the `logs/` directory
- Generated plots of rewards and episode lengths

## Model Parameters

The A2C agent is configured with the following parameters:
- Learning rate: 0.0007
- Number of steps: 5
- Discount factor (gamma): 0.99
- Policy network: MLP (Multi-Layer Perceptron)

You can modify these parameters in the `train_a2c.py` script to experiment with different configurations. 