import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import os
from collections import deque
import random

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.zeros((out_features, in_features)))
        self.weight_sigma = nn.Parameter(torch.zeros((out_features, in_features)))
        self.register_buffer('weight_epsilon', torch.zeros((out_features, in_features)))
        
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor with residual connections
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Actor head with noisy layers for exploration
        self.actor = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, n_actions)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # Critic head with value bounds
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, NoisyLinear):
                # NoisyLinear layers are already initialized in their constructor
                continue
    
    def reset_noise(self):
        for module in self.actor.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        features = self.feature_layer(x)
        
        # Get logits and apply temperature scaling
        logits = self.actor(features)
        scaled_logits = logits / self.temperature
        
        # Compute log probabilities with numerical stability
        log_probs = torch.log_softmax(scaled_logits, dim=-1)
        
        state_value = self.critic(features)
        return log_probs, state_value

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x):
        return x + self.layers(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001
        self.epsilon = 1e-6  # Small constant to prevent zero probabilities
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        if self.size < batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample indices and calculate importance weights
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )
    
    def update_priorities(self, indices, td_errors):
        priorities = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = priorities

class A2CAgent:
    def __init__(self, state_dim, n_actions, hidden_dim=256, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gae_lambda = 0.95
        self.n_steps = 5
        self.value_clip = 0.2
        self.max_grad_norm = 0.5
        
        # Enhanced exploration parameters
        self.entropy_coef = 0.2
        self.min_entropy = 0.8
        self.exploration_bonus = 0.05
        self.curiosity_weight = 0.2
        
        # Initialize reward statistics
        self.reward_scaling = 1.0
        self.reward_running_mean = 0.0
        self.reward_running_std = 1.0
        self.reward_alpha = 0.001
        self.reward_count = 0
        self.min_reward_std = 0.1
        
        # Potential-based reward shaping
        self.potential_scale = 20.0
        self.velocity_scale = 5.0
        self.position_scale = 10.0
        
        # Momentum-based exploration
        self.momentum_scale = 2.0
        self.momentum_threshold = 0.02
        self.last_velocity = 0
        self.velocity_momentum = 0
        self.momentum_decay = 0.95
        
        # Experience replay parameters
        self.replay_buffer = PrioritizedReplayBuffer()
        self.batch_size = 64
        self.min_replay_size = 1000
        self.replay_ratio = 4  # Number of replay updates per environment step
        
        # Model saving parameters
        self.best_score = float('-inf')
        self.best_position = float('-inf')
        self.save_dir = 'saved_models'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Initialize actor-critic network
        self.network = ActorCritic(state_dim, n_actions, hidden_dim).to(self.device)
        
        # Separate optimizers with different learning rates
        actor_params = list(self.network.feature_layer.parameters()) + list(self.network.actor.parameters()) + [self.network.temperature]
        critic_params = list(self.network.feature_layer.parameters()) + list(self.network.critic.parameters())
        
        self.actor_optimizer = optim.Adam(actor_params, lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr*2, eps=1e-5)
        
        # Learning rate scheduling with warmup
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.actor_optimizer, T_0=1000, T_mult=2, eta_min=lr*0.1
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.critic_optimizer, T_0=1000, T_mult=2, eta_min=lr*0.2
        )
        
        # Initialize memory
        self.clear_memory()
        
        # Performance tracking
        self.best_reward = float('-inf')
        self.best_position = float('-inf')
        self.running_reward = 0
        self.episode_count = 0
        self.visited_states = set()
        
        # State-dependent exploration
        self.state_visitation_counts = {}
        self.state_values = {}
        self.novelty_threshold = 10
    
    def compute_potential(self, state):
        """Compute potential for reward shaping with enhanced exploration signals"""
        position, velocity = state[0], state[1]
        
        # Update momentum-based exploration
        self.velocity_momentum = self.momentum_decay * self.velocity_momentum + (1 - self.momentum_decay) * abs(velocity - self.last_velocity)
        self.last_velocity = velocity
        
        # Height-based potential (higher positions have higher potential)
        height_potential = -np.cos(3 * position) * self.position_scale
        
        # Velocity-based potential (higher velocity magnitude has higher potential)
        velocity_potential = (abs(velocity) ** 2) * self.velocity_scale
        
        # Momentum bonus (encourage consistent acceleration)
        momentum_bonus = self.momentum_scale * self.velocity_momentum if abs(velocity) > self.momentum_threshold else 0
        
        # Goal proximity potential with progressive scaling
        distance_to_goal = abs(position - 0.5)
        goal_proximity = np.exp(-3 * distance_to_goal) * (1 + max(0, 0.5 - distance_to_goal) * 2)
        
        # Combine all potentials
        total_potential = (
            height_potential +
            velocity_potential +
            momentum_bonus +
            goal_proximity * self.potential_scale
        )
        
        return total_potential
    
    def normalize_reward(self, reward):
        """Numerically stable reward normalization with clipping"""
        self.reward_count += 1
        delta = reward - self.reward_running_mean
        
        # Update mean
        self.reward_running_mean += delta / self.reward_count
        
        # Update variance using Welford's online algorithm
        if self.reward_count > 1:
            delta2 = reward - self.reward_running_mean
            self.reward_running_std = np.sqrt(
                max(
                    self.min_reward_std,
                    (self.reward_running_std ** 2 * (self.reward_count - 2) + delta * delta2) / (self.reward_count - 1)
                )
            )
        
        # Normalize reward with clipping
        normalized_reward = (reward - self.reward_running_mean) / self.reward_running_std
        return float(np.clip(normalized_reward, -10.0, 10.0))
    
    def get_action_probs(self, logits):
        """Get action probabilities from logits with numerical stability"""
        return torch.softmax(logits, dim=-1)
    
    def choose_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            log_probs, _ = self.network(state)
            probs = self.get_action_probs(log_probs)
            
            # State-dependent exploration
            state_key = tuple(np.round(state.cpu().numpy(), 2))
            visit_count = self.state_visitation_counts.get(state_key, 0)
            
            # Progressive temperature scaling based on visit count and position
            position = state[0].item()
            base_temp = 2.0 - min(visit_count, self.novelty_threshold) * 0.1
            
            # Increase temperature for states closer to edges
            edge_factor = 1.0 + 0.5 * (abs(position + 1.2) < 0.1 or abs(position - 0.5) < 0.1)
            temperature = base_temp * edge_factor
            
            # Apply temperature scaling
            probs = probs ** (1 / temperature)
            probs = probs / probs.sum()  # Renormalize
            
            try:
                dist = Categorical(probs)
                action = dist.sample()
                
                # Update state visitation count
                self.state_visitation_counts[state_key] = visit_count + 1
                
                return action.item()
            except ValueError:
                return np.random.randint(3)  # Fallback to random action
    
    def store_transition(self, state, action, reward, next_state, done):
        # Store in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Store in episode memory
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        # Update running reward
        self.running_reward = (1 - self.reward_alpha) * self.running_reward + self.reward_alpha * reward
        
        # Update best metrics
        self.best_reward = max(self.best_reward, self.running_reward)
        if isinstance(next_state, (list, np.ndarray)):
            position = next_state[0]
            self.best_position = max(self.best_position, position)
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        running_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            running_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * running_gae
            advantages[t] = running_gae
            returns[t] = advantages[t] + values[t]
            
        return advantages, returns
    
    def normalize_advantages(self, advantages):
        """Normalize advantages with improved numerical stability"""
        mean = advantages.mean()
        std = advantages.std() + 1e-8
        normalized = (advantages - mean) / std
        return torch.clamp(normalized, -3, 3)
    
    def learn(self):
        actor_losses = []
        critic_losses = []
        
        # Learn from current episode
        if len(self.states) > 0:
            actor_loss, critic_loss = self._learn_from_episode()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        
        # Learn from replay buffer
        if self.replay_buffer.size >= self.min_replay_size:
            for _ in range(self.replay_ratio):
                batch = self.replay_buffer.sample(self.batch_size)
                if batch is not None:
                    actor_loss, critic_loss = self._learn_from_batch(*batch)
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
        
        # Clear episode memory
        self.clear_memory()
        
        # Return average losses
        return (
            np.mean(actor_losses) if actor_losses else 0,
            np.mean(critic_losses) if critic_losses else 0
        )
    
    def _learn_from_episode(self):
        # Convert episode memory to tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        return self._update_networks(states, actions, rewards, next_states, dones)
    
    def _learn_from_batch(self, states, actions, rewards, next_states, dones, indices, weights):
        # Convert batch to tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Get current values and update priorities
        with torch.no_grad():
            _, current_values = self.network(states)
            _, next_values = self.network(next_states)
            td_errors = rewards + self.gamma * next_values.squeeze() * (1 - dones) - current_values.squeeze()
            self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())
        
        return self._update_networks(states, actions, rewards, next_states, dones, weights)
    
    def _update_networks(self, states, actions, rewards, next_states, dones, weights=None):
        # Reset noisy layers
        self.network.reset_noise()
        
        # Get current logits and values
        log_probs, state_values = self.network(states)
        probs = torch.softmax(log_probs, dim=-1).clone()
        
        with torch.no_grad():
            _, next_state_values = self.network(next_states)
            advantages, returns = self.compute_gae(rewards, state_values.squeeze().detach(), next_state_values.squeeze(), dones)
            advantages = self.normalize_advantages(advantages)
            
            # Apply importance weights if provided
            if weights is not None:
                advantages = advantages * weights
                returns = returns * weights
        
        # Get action probabilities and calculate entropy
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        # Calculate actor loss
        actor_loss = -(action_log_probs * advantages).mean()
        entropy_loss = -self.entropy_coef * entropy
        if entropy < self.min_entropy:
            entropy_loss *= 2.0
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_total_loss = actor_loss + entropy_loss
        actor_total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.network.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Calculate critic loss with fresh forward pass
        _, new_state_values = self.network(states)
        value_pred = new_state_values.squeeze()
        value_target = returns.clone().detach()
        value_pred_clipped = value_pred.clone() + torch.clamp(value_pred - value_target, -self.value_clip, self.value_clip)
        value_losses = (value_pred - value_target).pow(2)
        value_losses_clipped = (value_pred_clipped - value_target).pow(2)
        critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Update learning rates
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
    def save_model(self, filename):
        state = {
            'network_state': self.network.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'best_score': self.best_score,
            'best_position': self.best_position,
            'reward_stats': {
                'mean': self.reward_running_mean,
                'std': self.reward_running_std,
                'count': self.reward_count
            },
            'state_visitation': self.state_visitation_counts
        }
        torch.save(state, os.path.join(self.save_dir, filename))
        
    def load_model(self, filename):
        state = torch.load(os.path.join(self.save_dir, filename))
        self.network.load_state_dict(state['network_state'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.actor_scheduler.load_state_dict(state['actor_scheduler'])
        self.critic_scheduler.load_state_dict(state['critic_scheduler'])
        self.best_score = state['best_score']
        self.best_position = state['best_position']
        stats = state['reward_stats']
        self.reward_running_mean = stats['mean']
        self.reward_running_std = stats['std']
        self.reward_count = stats['count']
        if 'state_visitation' in state:
            self.state_visitation_counts = state['state_visitation']
        
    def evaluate(self, env, n_episodes=10, render=False):
        """Evaluate the current model for n episodes"""
        self.network.eval()  # Set to evaluation mode
        scores = []
        max_positions = []
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_score = 0
            max_position = float('-inf')
            done = False
            
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    log_probs, _ = self.network(state_tensor)
                    probs = self.get_action_probs(log_probs)
                    action = torch.argmax(probs).item()  # Use greedy action selection during evaluation
                
                next_state, reward, done, _ = env.step(action)
                if render:
                    env.render()
                
                episode_score += reward
                max_position = max(max_position, next_state[0])
                state = next_state
            
            scores.append(episode_score)
            max_positions.append(max_position)
        
        self.network.train()  # Set back to training mode
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_position': np.mean(max_positions),
            'max_position': max(max_positions),
            'success_rate': sum(1 for pos in max_positions if pos >= 0.5) / n_episodes
        } 