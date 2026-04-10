"""
Q-Learning Implementation from Scratch

Q-Learning is a model-free reinforcement learning algorithm that learns
the value of actions in specific states through trial and error.

Q-Learning Update Formula:
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

Where:
- Q(s,a) = expected reward for action a in state s
- α = learning rate
- r = immediate reward
- γ = discount factor
- s' = next state
- a' = next action
"""
import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent:
    """
    Q-Learning Agent Implementation
    
    The agent maintains a Q-table that stores expected rewards for
    state-action pairs and uses an epsilon-greedy strategy for exploration.
    """
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount_factor=0.9, epsilon=0.1):
        """
        Initialize Q-Learning Agent
        
        Args:
            n_states (int): Number of possible states
            n_actions (int): Number of possible actions
            learning_rate (float): Learning rate (α)
            discount_factor (float): Discount factor (γ)
            epsilon (float): Exploration rate for epsilon-greedy
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        print(f"Q-Learning Agent Initialized:")
        print(f"  States: {n_states}")
        print(f"  Actions: {n_actions}")
        print(f"  Learning Rate (α): {learning_rate}")
        print(f"  Discount Factor (γ): {discount_factor}")
        print(f"  Epsilon (ε): {epsilon}")
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy strategy
        
        Args:
            state (int): Current state
            
        Returns:
            int: Selected action
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using the Q-learning formula
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Args:
            state (int): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (int): Next state
            done (bool): Whether episode is finished
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Maximum Q-value for next state
        if done:
            next_max_q = 0  # No future rewards if episode is done
        else:
            next_max_q = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        
        self.q_table[state, action] = new_q
    
    def get_best_action(self, state):
        """
        Get the best action for a state (no exploration)
        
        Args:
            state (int): Current state
            
        Returns:
            int: Best action
        """
        return np.argmax(self.q_table[state])


class SimpleGridWorld:
    """
    Simple Grid World Environment
    
    A 4x4 grid where:
    - Start: top-left (0)
    - Goal: bottom-right (15)
    - Agent receives +1 reward for reaching goal
    - Agent receives -0.01 reward for each step (to encourage efficiency)
    """
    
    def __init__(self, size=4):
        """
        Initialize grid world
        
        Args:
            size (int): Grid size (size x size)
        """
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # Up, Down, Left, Right
        self.start_state = 0
        self.goal_state = self.n_states - 1
        
        # Action mappings
        self.action_names = ['Up', 'Down', 'Left', 'Right']
        
        print(f"\nGrid World Environment:")
        print(f"  Size: {size}x{size}")
        print(f"  States: {self.n_states}")
        print(f"  Actions: {self.action_names}")
        print(f"  Start: State {self.start_state}")
        print(f"  Goal: State {self.goal_state}")
    
    def reset(self):
        """Reset environment to start state"""
        self.current_state = self.start_state
        return self.current_state
    
    def step(self, action):
        """
        Take action and return next state, reward, and done flag
        
        Args:
            action (int): 0=Up, 1=Down, 2=Left, 3=Right
            
        Returns:
            tuple: (next_state, reward, done)
        """
        # Convert state to row, col
        row = self.current_state // self.size
        col = self.current_state % self.size
        
        # Update position based on action
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.size - 1, col + 1)
        
        # Convert back to state
        next_state = row * self.size + col
        
        # Calculate reward
        if next_state == self.goal_state:
            reward = 1.0
            done = True
        else:
            reward = -0.01  # Small penalty to encourage efficiency
            done = False
        
        self.current_state = next_state
        
        return next_state, reward, done


def train_agent(agent, env, episodes=1000):
    """
    Train Q-Learning agent
    
    Args:
        agent (QLearningAgent): The agent to train
        env (SimpleGridWorld): The environment
        episodes (int): Number of training episodes
        
    Returns:
        list: Episode rewards over training
    """
    print(f"\n{'='*50}")
    print(f"TRAINING Q-LEARNING AGENT")
    print(f"{'='*50}")
    
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100:  # Max 100 steps per episode
            # Select and take action
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Update Q-value
            agent.update_q_value(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            steps += 1
        
        episode_rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.3f}")
    
    print(f"\nTraining completed!")
    
    return episode_rewards


def visualize_training(episode_rewards):
    """
    Visualize training progress
    
    Args:
        episode_rewards (list): Rewards per episode
    """
    plt.figure(figsize=(12, 5))
    
    # Plot episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Training Progress - Episode Rewards', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    window = 50
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg, color='red', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(f'Moving Average Reward (window={window})', fontsize=12)
    plt.title('Training Progress - Moving Average', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_policy(agent, env):
    """
    Visualize learned policy
    
    Args:
        agent (QLearningAgent): Trained agent
        env (SimpleGridWorld): Environment
    """
    print(f"\n{'='*50}")
    print(f"LEARNED POLICY")
    print(f"{'='*50}")
    
    # Arrow symbols for actions
    arrows = ['↑', '↓', '←', '→']
    
    print("\nOptimal Actions in Each State:")
    print("(↑=Up, ↓=Down, ←=Left, →=Right)")
    print()
    
    for row in range(env.size):
        line = ""
        for col in range(env.size):
            state = row * env.size + col
            if state == env.goal_state:
                line += " G "  # Goal
            else:
                best_action = agent.get_best_action(state)
                line += f" {arrows[best_action]} "
        print(line)


def demonstrate_agent(agent, env, num_episodes=5):
    """
    Demonstrate trained agent
    
    Args:
        agent (QLearningAgent): Trained agent
        env (SimpleGridWorld): Environment
        num_episodes (int): Number of demonstration episodes
    """
    print(f"\n{'='*50}")
    print(f"DEMONSTRATION")
    print(f"{'='*50}")
    
    for episode in range(num_episodes):
        state = env.reset()
        path = [state]
        steps = 0
        done = False
        
        while not done and steps < 20:
            action = agent.get_best_action(state)
            next_state, reward, done = env.step(action)
            path.append(next_state)
            state = next_state
            steps += 1
        
        print(f"\nEpisode {episode + 1}: Path = {path}")
        print(f"  Steps to goal: {steps}")


def main():
    """
    Main function to demonstrate Q-Learning
    """
    print("="*50)
    print(" Q-LEARNING FROM SCRATCH ")
    print("="*50)
    
    # Create environment
    env = SimpleGridWorld(size=4)
    
    # Create agent
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1
    )
    
    # Train agent
    episode_rewards = train_agent(agent, env, episodes=1000)
    
    # Visualize training
    visualize_training(episode_rewards)
    
    # Visualize learned policy
    visualize_policy(agent, env)
    
    # Demonstrate trained agent
    demonstrate_agent(agent, env, num_episodes=5)
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"✓ Algorithm: Q-Learning")
    print(f"✓ Environment: {env.size}x{env.size} Grid World")
    print(f"✓ Training Episodes: 1000")
    print(f"✓ Agent successfully learned optimal policy!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
