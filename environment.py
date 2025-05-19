import gymnasium as gym
import numpy as np
from collections import deque

class MountainCarEnv:
    def __init__(self, render_mode=None, curriculum_level=0):
        self.env = gym.make('MountainCar-v0', render_mode=render_mode)
        self.state_dim = self.env.observation_space.shape[0]  # 2 for MountainCar (position, velocity)
        self.n_actions = self.env.action_space.n  # 3 for MountainCar (left, nothing, right)
        
        # Curriculum learning parameters
        self.curriculum_level = curriculum_level
        self.max_curriculum_level = 10
        self.success_threshold = 0.25  # Increased from 0.2
        self.success_window = deque(maxlen=100)  # Increased from 50
        
        # Modified environment parameters
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        
        # Track previous states for oscillation detection
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=10)
        self.last_velocity = 0
        self.max_position_achieved = -np.inf
        
        # Potential-based reward shaping parameters
        self.gamma = 0.99  # Discount factor for potential shaping
        self.position_scale = 1.0
        self.velocity_scale = 0.5
        self.height_scale = 2.0
        
        # Curiosity and exploration parameters
        self.state_visitation = {}  # Track state visitation counts
        self.state_resolution = 20  # Number of bins for state discretization
        self.novelty_scale = 0.5
        self.min_novelty_reward = 0.1
        
        # Initialize state bins for curiosity
        self.position_bins = np.linspace(-1.2, 0.6, self.state_resolution)
        self.velocity_bins = np.linspace(-0.07, 0.07, self.state_resolution)
        
    def get_curriculum_start_position(self):
        """Get starting position based on curriculum level with more gradual progression"""
        if self.curriculum_level == 0:
            # Start close to the goal with high velocity
            return 0.4, 0.07
        elif self.curriculum_level < 5:
            # More gradual position changes
            pos = 0.4 - 0.05 * self.curriculum_level
            vel = 0.07 - 0.005 * self.curriculum_level
            return pos, vel
        else:
            # More controlled random starts with progressive difficulty
            level_factor = (self.curriculum_level - 5) / 5  # 0 to 1 for levels 5-10
            min_pos = -0.6 - 0.4 * level_factor  # Gradually decrease to -1.0
            max_pos = 0.4 - 0.2 * level_factor  # Gradually decrease to 0.2
            return np.random.uniform(min_pos, max_pos), 0.02
    
    def get_state_potential(self, state):
        """Calculate potential function for a state"""
        position, velocity = state
        
        # Height-based potential (higher positions have higher potential)
        height = np.sin(3 * position)  # Approximate height in the valley
        height_potential = height * self.height_scale
        
        # Position-based potential (closer to goal has higher potential)
        position_potential = (position - self.min_position) / (self.goal_position - self.min_position)
        position_potential *= self.position_scale
        
        # Velocity-based potential (higher velocity has higher potential when going in the right direction)
        velocity_potential = 0
        if position <= 0:  # In the valley
            velocity_potential = abs(velocity) * self.velocity_scale  # Any velocity is good for building momentum
        else:  # Going uphill
            velocity_potential = velocity * self.velocity_scale  # Only positive velocity is good
        
        return height_potential + position_potential + velocity_potential
    
    def get_discretized_state(self, state):
        """Convert continuous state to discrete for curiosity tracking"""
        position, velocity = state
        position_bin = np.digitize(position, self.position_bins)
        velocity_bin = np.digitize(velocity, self.velocity_bins)
        return (position_bin, velocity_bin)
    
    def get_novelty_reward(self, state):
        """Calculate novelty reward based on state visitation"""
        discrete_state = self.get_discretized_state(state)
        visit_count = self.state_visitation.get(discrete_state, 0)
        
        # Novelty reward decreases with more visits but never goes below minimum
        novelty_reward = max(self.novelty_scale / (visit_count + 1), self.min_novelty_reward)
        
        # Update visitation count
        self.state_visitation[discrete_state] = visit_count + 1
        
        return novelty_reward
    
    def detect_oscillation(self):
        """Detect if the agent is building momentum through oscillation"""
        if len(self.position_history) < 10:
            return False, 0
        
        # Check for alternating velocity directions
        vel_changes = sum(1 for i in range(1, len(self.velocity_history))
                         if self.velocity_history[i] * self.velocity_history[i-1] < 0)
        
        # Check for increasing amplitude
        pos_amplitude = max(self.position_history) - min(self.position_history)
        vel_amplitude = max(map(abs, self.velocity_history))
        
        # Check for effective oscillation pattern
        is_oscillating = vel_changes >= 4
        
        # Quality metric considers both amplitude and frequency
        quality = 0
        if is_oscillating:
            # Higher quality for larger amplitudes and higher velocities
            quality = pos_amplitude * vel_amplitude
            # Bonus for optimal oscillation frequency (4-6 changes is ideal)
            if 4 <= vel_changes <= 6:
                quality *= 1.5
        
        return is_oscillating, quality
    
    def reset(self):
        state, _ = self.env.reset()
        if self.curriculum_level < self.max_curriculum_level:
            pos, vel = self.get_curriculum_start_position()
            self.env.unwrapped.state = np.array([pos, vel])
            state = self.env.unwrapped.state
        
        # Reset tracking variables
        self.position_history.clear()
        self.velocity_history.clear()
        self.last_velocity = abs(state[1])
        self.max_position_achieved = state[0]
        
        return state
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Update state history
        position = next_state[0]
        velocity = next_state[1]
        abs_velocity = abs(velocity)
        
        self.position_history.append(position)
        self.velocity_history.append(velocity)
        self.max_position_achieved = max(self.max_position_achieved, position)
        
        # Calculate potential-based reward
        current_potential = self.get_state_potential(next_state)
        prev_potential = self.get_state_potential([self.position_history[-2], self.velocity_history[-2]] if len(self.position_history) > 1 else next_state)
        shaping_reward = self.gamma * current_potential - prev_potential
        
        # Get novelty reward
        novelty_reward = self.get_novelty_reward(next_state)
        
        # Base reward
        reward = -1.0
        
        # Goal achievement reward
        if position >= self.goal_position:
            reward = 100.0
            done = True
        else:
            # Progressive reward shaping based on curriculum level
            if self.curriculum_level < 5:
                # Early curriculum levels: focus on basic progress
                progress_reward = (position - self.min_position) / (self.goal_position - self.min_position) * 3
                velocity_reward = (velocity / self.max_speed * 2) if position > 0 and velocity > 0 else 0
                momentum_reward = 0.5 if abs_velocity > self.last_velocity else 0
                
                reward += progress_reward + velocity_reward + momentum_reward
            else:
                # Later curriculum levels: focus on advanced strategies
                is_oscillating, oscillation_quality = self.detect_oscillation()
                
                # Strategic rewards
                if position < 0:  # In the valley
                    if abs_velocity > 0.04:  # Building momentum
                        reward += 0.4
                    if is_oscillating:  # Good oscillation pattern
                        reward += oscillation_quality * 0.5
                else:  # Going uphill
                    if velocity > 0:  # Moving toward goal
                        reward += (velocity / self.max_speed) * 2
                    if position > self.max_position_achieved - 0.01:  # New height achieved
                        reward += 0.5
            
            # Add potential-based and novelty rewards
            reward += shaping_reward + novelty_reward
        
        # Update velocity tracking
        self.last_velocity = abs_velocity
        
        # Update curriculum
        if done:
            self.success_window.append(1 if position >= self.goal_position else 0)
            success_rate = sum(self.success_window) / len(self.success_window)
            
            if len(self.success_window) >= 100 and success_rate >= self.success_threshold:  # Increased requirements
                self.curriculum_level = min(self.curriculum_level + 1, self.max_curriculum_level)
                self.success_window.clear()
        
        return next_state, reward, done, info
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
        
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space 