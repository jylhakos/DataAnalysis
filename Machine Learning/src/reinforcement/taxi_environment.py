"""
Q-Learning with Gymnasium Taxi Environment

This script demonstrates Q-Learning on the Taxi-v3 environment from Gymnasium.

Environment Description:
- 5x5 grid world
- 4 passenger locations (R, G, Y, B)
- Agent must pick up passenger and drop off at correct destination
- Actions: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff

Rewards:
- +20 for successful dropoff
- -10 for illegal pickup/dropoff
- -1 per step

State Space: 500 states (25 taxi positions × 5 passenger locations × 4 destinations)
Action Space: 6 actions
"""
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict


class TaxiQLearningAgent:
    """
    Q-Learning Agent for Taxi Environment
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, 
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize agent
        
        Args:
            env: Gymnasium environment
            learning_rate (float): Learning rate (α)
            discount_factor (float): Discount factor (γ)
            epsilon_start (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Epsilon decay rate
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        print("Taxi Q-Learning Agent Initialized")
        print(f"  State Space: {env.observation_space.n}")
        print(f"  Action Space: {env.action_space.n}")
        print(f"  Learning Rate (α): {learning_rate}")
        print(f"  Discount Factor (γ): {discount_factor}")
        print(f"  Epsilon: {epsilon_start} → {epsilon_min}")
    
    def select_action(self, state):
        """
        Epsilon-greedy action selection
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning formula
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode finished
        """
        current_q = self.q_table[state][action]
        
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_taxi_agent(agent, env, episodes=10000, print_every=1000):
    """
    Train agent on Taxi environment
    
    Args:
        agent: TaxiQLearningAgent
        env: Gymnasium Taxi environment
        episodes: Number of training episodes
        print_every: Print progress every N episodes
        
    Returns:
        dict: Training statistics
    """
    print(f"\n{'='*60}")
    print(f"TRAINING ON TAXI ENVIRONMENT")
    print(f"{'='*60}")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            # Select and take action
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update Q-value
            agent.update_q_value(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            steps += 1
        
        # Track statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Check for successful delivery
        if total_reward > 0:
            success_count += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_length = np.mean(episode_lengths[-print_every:])
            success_rate = (success_count / print_every) * 100
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f} steps")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            success_count = 0
    
    print(f"\nTraining completed!")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def visualize_training_results(stats):
    """
    Visualize training statistics
    
    Args:
        stats (dict): Training statistics
    """
    episode_rewards = stats['episode_rewards']
    episode_lengths = stats['episode_lengths']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Moving average rewards
    window = 100
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    axes[0, 1].plot(moving_avg, color='red', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel(f'Moving Avg Reward (window={window})')
    axes[0, 1].set_title('Smoothed Performance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[1, 0].plot(episode_lengths, alpha=0.5, color='green')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward distribution (last 1000 episodes)
    axes[1, 1].hist(episode_rewards[-1000:], bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution (Last 1000 Episodes)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_trained_agent(agent, env, num_episodes=5):
    """
    Demonstrate trained agent
    
    Args:
        agent: Trained TaxiQLearningAgent
        env: Taxi environment
        num_episodes: Number of demonstration episodes
    """
    print(f"\n{'='*60}")
    print(f"DEMONSTRATION OF TRAINED AGENT")
    print(f"{'='*60}")
    
    action_names = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
    
    # Set epsilon to 0 for pure exploitation
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        
        print(f"\nEpisode {episode + 1}:")
        print(f"{'='*40}")
        
        while not done and not truncated and steps < 50:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Print action taken
            if steps <= 10:  # Print first 10 actions
                print(f"  Step {steps}: {action_names[action]} → Reward: {reward:+.0f}")
            
            state = next_state
        
        status = "SUCCESS" if total_reward > 0 else "FAILED"
        print(f"\n  Total Steps: {steps}")
        print(f"  Total Reward: {total_reward:+.0f}")
        print(f"  Status: {status}")
    
    # Restore epsilon
    agent.epsilon = original_epsilon


def evaluate_agent(agent, env, num_eval_episodes=100):
    """
    Evaluate agent performance
    
    Args:
        agent: Trained agent
        env: Environment
        num_eval_episodes: Number of evaluation episodes
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING AGENT PERFORMANCE")
    print(f"{'='*60}")
    
    # Set epsilon to 0 for evaluation (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    rewards = []
    lengths = []
    success_count = 0
    
    for _ in range(num_eval_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
        
        rewards.append(total_reward)
        lengths.append(steps)
        
        if total_reward > 0:
            success_count += 1
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    # Calculate metrics
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_length = np.mean(lengths)
    success_rate = (success_count / num_eval_episodes) * 100
    
    print(f"\nEvaluation Results ({num_eval_episodes} episodes):")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Average Steps: {avg_length:.1f}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'success_rate': success_rate
    }


def main():
    """
    Main function to demonstrate Q-Learning on Taxi environment
    """
    print("="*60)
    print(" Q-LEARNING WITH GYMNASIUM TAXI ENVIRONMENT ")
    print("="*60)
    
    # Create environment
    env = gym.make('Taxi-v3')
    
    print("\nEnvironment Information:")
    print(f"  Name: Taxi-v3")
    print(f"  Description: Pick up and drop off passengers in a 5x5 grid")
    print(f"  State Space: {env.observation_space.n} states")
    print(f"  Action Space: {env.action_space.n} actions")
    print(f"    0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff")
    
    # Create agent
    agent = TaxiQLearningAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Train agent
    stats = train_taxi_agent(agent, env, episodes=10000, print_every=2000)
    
    # Visualize training
    visualize_training_results(stats)
    
    # Evaluate agent
    eval_metrics = evaluate_agent(agent, env, num_eval_episodes=100)
    
    # Demonstrate agent
    demonstrate_trained_agent(agent, env, num_episodes=3)
    
    # Close environment
    env.close()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Algorithm: Q-Learning")
    print(f"✓ Environment: Taxi-v3")
    print(f"✓ Training Episodes: 10,000")
    print(f"✓ Final Success Rate: {eval_metrics['success_rate']:.1f}%")
    print(f"✓ Average Steps to Complete: {eval_metrics['avg_length']:.1f}")
    print(f"✓ Agent successfully learned to navigate and complete taxi tasks!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
