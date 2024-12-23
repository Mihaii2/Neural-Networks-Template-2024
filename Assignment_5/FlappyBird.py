import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from PIL import Image
import os
from datetime import datetime
import yaml


class DQN(nn.Module):
    def __init__(self, input_channels=4, n_actions=2):
        super(DQN, self).__init__()
        # Using LeakyReLU with a default negative slope of 0.01
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.01)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc_layers(x)


class FrameStack:
    def __init__(self, size=4):
        self.size = size
        self.frames = deque(maxlen=size)

    def push(self, frame):
        self.frames.append(frame)

    def get_state(self):
        # If we don't have enough frames, duplicate the last frame
        while len(self.frames) < self.size:
            if len(self.frames) > 0:
                self.frames.append(self.frames[-1])
            else:
                # Create a zero tensor with same shape as expected frame: [1, 1, 84, 84]
                zero_frame = torch.zeros(1, 1, 84, 84)
                self.frames.append(zero_frame)

        # Stack frames along channel dimension
        return torch.cat(list(self.frames), dim=1)  # Will give [1, 4, 84, 84]


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.cat(state),
                torch.tensor(action),
                torch.tensor(reward),
                torch.cat(next_state),
                torch.tensor(done))

    def __len__(self):
        return len(self.buffer)


def preprocess_image(image):
    """
    Preprocess the image by converting it to grayscale, cropping the bottom, and resizing it.
    Returns the image as a PyTorch tensor.
    """
    # Convert to PIL Image for processing
    pil_image = Image.fromarray(image)

    # Get image dimensions
    width, height = pil_image.size

    # Crop the bottom portion (remove approximately 20% from bottom)
    crop_height = int(height * 0.8)  # Keep top 80%
    cropped_image = pil_image.crop((0, 0, width, crop_height))

    # Convert to grayscale
    grayscale_image = cropped_image.convert("L")

    # Resize the image to a fixed size (84x84)
    resized_image = grayscale_image.resize((84, 84))

    # Convert to numpy array and normalize
    preprocessed_image = np.array(resized_image) / 255.0

    # Convert to PyTorch tensor and add batch and channel dimensions
    tensor_image = torch.FloatTensor(preprocessed_image).unsqueeze(0).unsqueeze(0)

    return tensor_image


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_channels=4).to(self.device)  # Changed to 4 input channels
        self.target_net = DQN(input_channels=4).to(self.device)  # Changed to 4 input channels
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.frame_stack = FrameStack(size=4)  # Add frame stack

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.memory = ReplayBuffer()

        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon_start = 0.5
        self.epsilon_end = 0.05
        self.epsilon_decay = 300
        self.current_epsilon = self.epsilon_start
        self.target_update = 1000
        self.frame_skip = 3

        self.steps_done = 0

    def select_action(self, state, training=True):
        if self.steps_done % self.epsilon_decay == 0:
            decay_amount = 0.001
            self.current_epsilon = max(self.epsilon_end, self.current_epsilon - decay_amount)

        if training and random.random() < self.current_epsilon:
            return random.randint(0, 1)

        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

    def update_hyperparameters(self):
        """Update hyperparameters based on the YAML configuration"""
        try:
            with open('hyperparameter_updates.yaml', 'r') as file:
                config = yaml.safe_load(file)

            if config['status'] == 'Update':
                print("\nUpdating hyperparameters...")
                updates = config['updates']

                # Update learning rate
                if 'learning_rate' in updates:
                    lr_update = updates['learning_rate']
                    current_lr = self.optimizer.param_groups[0]['lr']

                    if lr_update['action'] == 'add':
                        new_lr = current_lr + lr_update['value']
                    elif lr_update['action'] == 'sub':
                        new_lr = current_lr - lr_update['value']
                    elif lr_update['action'] == 'mul':
                        new_lr = current_lr * lr_update['value']

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"Learning rate updated: {current_lr:.4f} -> {new_lr:.7f}")

                # Update current epsilon
                if 'epsilon' in updates:
                    eps_update = updates['epsilon']
                    old_epsilon = self.current_epsilon
                    if eps_update['action'] == 'add':
                        self.current_epsilon = min(1.0, self.current_epsilon + eps_update['value'])
                    elif eps_update['action'] == 'sub':
                        self.current_epsilon = max(self.epsilon_end, self.current_epsilon - eps_update['value'])
                    print(f"Current epsilon updated: {old_epsilon:.4f} -> {self.current_epsilon:.4f}")

                # Update epsilon decay
                if 'epsilon_decay' in updates:
                    decay_update = updates['epsilon_decay']
                    old_decay = self.epsilon_decay
                    if decay_update['action'] == 'mul':
                        self.epsilon_decay = int(self.epsilon_decay * decay_update['value'])
                    print(f"Epsilon decay updated: {old_decay} -> {self.epsilon_decay}")

                # Set status to "Ignore" after applying updates
                config['status'] = 'Ignore'
                with open('hyperparameter_updates.yaml', 'w') as file:
                    yaml.dump(config, file)

                print("Hyperparameter updates complete\n")

        except Exception as e:
            print(f"Error updating hyperparameters: {str(e)}")

    def load_model(self, episode_number):
        """Load a previously saved model and handle architecture changes"""
        model_path = os.path.join('models', f'dqn_episode_{episode_number}.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)

            # Handle the case where we're loading a single-channel model into a four-channel model
            old_state_dict = checkpoint['model_state_dict']
            new_state_dict = self.policy_net.state_dict()

            # Special handling for the first conv layer
            if 'conv_layers.0.weight' in old_state_dict:
                old_weights = old_state_dict['conv_layers.0.weight']
                if old_weights.size(1) == 1 and new_state_dict['conv_layers.0.weight'].size(1) == 4:
                    # Duplicate the single channel weights across all 4 channels
                    new_weights = old_weights.repeat(1, 4, 1, 1)
                    old_state_dict['conv_layers.0.weight'] = new_weights

            # Load the modified state dict
            self.policy_net.load_state_dict(old_state_dict)
            self.target_net.load_state_dict(old_state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_done = episode_number * 1000  # Approximate steps based on episode
            print(f"Successfully loaded and adapted model from episode {episode_number}")
            return checkpoint['episode'], checkpoint['reward']
        else:
            raise FileNotFoundError(f"No model found at {model_path}")

    def train(self, num_episodes=2000000, start_episode=None, load_existing=True):
        """Modified train method with optional model loading"""
        os.makedirs('models', exist_ok=True)
        os.makedirs('images', exist_ok=True)

        rewards_history = deque(maxlen=100)
        best_reward = float('-inf')

        # Load previous model if specified and load_existing is True
        if start_episode is not None and load_existing:
            try:
                episode_num, last_reward = self.load_model(start_episode)
                best_reward = last_reward
                start_episode = episode_num
                print("Continuing training from episode", start_episode)
            except FileNotFoundError:
                print(f"No existing model found for episode {start_episode}, starting fresh")
                start_episode = 0
        else:
            print("Starting fresh training run")
            start_episode = 0

        for episode in range(start_episode, num_episodes):
            state, _ = self.env.reset()
            frame = preprocess_image(self.env.render())

            # Initialize frame stack with first frame
            self.frame_stack = FrameStack(size=4)
            for _ in range(4):
                self.frame_stack.push(frame)
            state = self.frame_stack.get_state()

            episode_reward = 0
            done = False

            while not done:
                action = self.select_action(state)

                # Frame skipping
                skip_reward = 0
                for _ in range(self.frame_skip):
                    next_state, reward, done, truncated, _ = self.env.step(action)
                    skip_reward += reward
                    if done:
                        break

                next_frame = preprocess_image(self.env.render())
                self.frame_stack.push(next_frame)
                next_state = self.frame_stack.get_state()

                # Store transition in memory
                self.memory.push(state, action, skip_reward, next_state, done)

                # Optimize model
                self.optimize_model()

                state = next_state
                episode_reward += skip_reward
                self.steps_done += 1

                # Update target network
                if self.steps_done % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            rewards_history.append(episode_reward)
            avg_reward = np.mean(rewards_history)

            # Save model periodically
            if episode % 1000 == 0:
                model_path = os.path.join('models', f'dqn_episode_{episode}.pth')
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'reward': episode_reward,
                }, model_path)
                print(f"Model saved to {model_path}")

            # Save model when we achieve best episode reward
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_model_path = os.path.join('models', 'dqn_best.pth')
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'reward': episode_reward,
                }, best_model_path)
                print(f"New best reward achieved: {best_reward:.2f}")

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Reward: {episode_reward:.2f} - "
                      f"Average Reward (100 ep): {avg_reward:.2f} - "
                      f"Best Reward: {best_reward:.2f} - "
                      f"Epsilon: {self.current_epsilon:.7f} - "
                      f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.7f}")

            self.update_hyperparameters()


def main():
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    agent = DQNAgent(env)
    agent.train(start_episode=None)
    env.close()


if __name__ == "__main__":
    main()