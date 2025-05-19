# Actor-Critic Advantage (A2C) RL Implementation

This project implements an A2C reinforcement learning agent to solve the CartPole environment from Gymnasium (formerly OpenAI Gym).

## Environment Description
CartPole is a classic control problem where a pole is attached to a cart that moves along a track. The goal is to prevent the pole from falling over by moving the cart left or right.

### State Space (4 dimensions):
- Cart Position
- Cart Velocity
- Pole Angle
- Pole Angular Velocity

### Action Space (2 actions):
- Push cart left
- Push cart right

### Reward:
- +1 for every timestep the pole remains upright
- Episode ends when:
  - Pole angle is more than ±12 degrees
  - Cart position is more than ±2.4 units
  - Episode length reaches 500 timesteps

## Setup
1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training:
```bash
python train.py
```

## Project Structure
- `train.py`: Main training loop
- `a2c_agent.py`: Implementation of the A2C agent
- `environment.py`: Environment wrapper and utilities 