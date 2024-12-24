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
import gc


class DQN(nn.Module):
    def __init__(self, input_channels=4, n_actions=2):
        super(DQN, self).__init__()
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
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class FrameStack:
    def __init__(self, size=4):
        self.size = size
        self.frames = deque(maxlen=size)

    def push(self, frame):
        # Ensure frame is detached from computation graph and on CPU
        if torch.is_tensor(frame):
            frame = frame.cpu().detach()
        self.frames.append(frame)

    def get_state(self):
        while len(self.frames) < self.size:
            if len(self.frames) > 0:
                self.frames.append(self.frames[-1])
            else:
                zero_frame = torch.zeros(1, 1, 84, 84)
                self.frames.append(zero_frame)

        # Stack frames and detach to prevent memory leaks
        stacked = torch.cat(list(self.frames), dim=1)
        return stacked.detach()

    def clear(self):
        self.frames.clear()


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Move tensors to CPU and detach from computation graph
        if torch.is_tensor(state):
            state = state.cpu().detach()
        if torch.is_tensor(next_state):
            next_state = next_state.cpu().detach()

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        # Stack all states efficiently
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


def preprocess_image(image):
    pil_image = Image.fromarray(image)
    width, height = pil_image.size
    crop_height = int(height * 0.8)
    cropped_image = pil_image.crop((0, 0, width, crop_height))
    grayscale_image = cropped_image.convert("L")
    resized_image = grayscale_image.resize((84, 84))
    preprocessed_image = np.array(resized_image) / 255.0
    tensor_image = torch.FloatTensor(preprocessed_image).unsqueeze(0).unsqueeze(0)

    # Clean up PIL Images
    del pil_image
    del cropped_image
    del grayscale_image
    del resized_image

    return tensor_image


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_channels=4).to(self.device)
        self.target_net = DQN(input_channels=4).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.frame_stack = FrameStack(size=4)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.memory = ReplayBuffer(capacity=50000)

        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon_start = 0.2
        self.epsilon_end = 0.001
        self.epsilon_decay = 200
        self.current_epsilon = self.epsilon_start
        self.target_update = 1000
        self.frame_skip = 0

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
            action = q_values.max(1)[1].item()
            # Clean up
            del q_values
            return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample and move to device
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute loss
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            expected_q_values = rewards.unsqueeze(1) + (1 - dones.float().unsqueeze(1)) * self.gamma * next_q_values

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        loss = nn.MSELoss()(current_q_values, expected_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Clean up
        del states, actions, rewards, next_states, dones
        del next_actions, next_q_values, expected_q_values, current_q_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def clear_memory(self):
        self.memory.clear()
        self.frame_stack.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self, num_episodes=10000, start_episode=None, load_existing=True):
        os.makedirs('models', exist_ok=True)
        os.makedirs('images', exist_ok=True)

        rewards_history = deque(maxlen=100)
        best_reward = float('-inf')

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
            start_episode = 0

        try:
            for episode in range(start_episode, num_episodes):
                state, _ = self.env.reset()
                frame = preprocess_image(self.env.render())

                self.frame_stack = FrameStack(size=4)
                for _ in range(4):
                    self.frame_stack.push(frame)
                state = self.frame_stack.get_state()

                episode_reward = 0
                done = False

                while not done:
                    action = self.select_action(state)

                    # Take at least one step regardless of frame_skip
                    next_state, reward, done, truncated, _ = self.env.step(action)
                    skip_reward = reward

                    # Additional frame skips if frame_skip > 0
                    for _ in range(max(0, self.frame_skip - 1)):
                        if done:
                            break
                        next_state, reward, done, truncated, _ = self.env.step(action)
                        skip_reward += reward

                    next_frame = preprocess_image(self.env.render())
                    self.frame_stack.push(next_frame)
                    next_state = self.frame_stack.get_state()

                    self.memory.push(state, action, skip_reward, next_state, done)
                    self.optimize_model()

                    state = next_state
                    episode_reward += skip_reward
                    self.steps_done += 1

                    if self.steps_done % self.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                rewards_history.append(episode_reward)
                avg_reward = np.mean(rewards_history)

                # Periodic cleanup
                if episode % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Save model periodically
                if episode % 100 == 0:
                    self.clear_memory()  # Clear memory before saving
                    model_path = os.path.join('models', f'dqn_episode_{episode}.pth')
                    torch.save({
                        'episode': episode,
                        'model_state_dict': self.policy_net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'reward': episode_reward,
                    }, model_path)
                    print(f"Model saved at episode {episode}")

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_model_path = os.path.join('models', 'dqn_best.pth')
                    torch.save({
                        'episode': episode,
                        'model_state_dict': self.policy_net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'reward': episode_reward,
                    }, best_model_path)
                    print(f"New best model saved. Episode {episode} with reward {episode_reward:.2f}")

                if (episode + 1) % 100 == 0:
                    print(f"Episode {episode + 1}/{num_episodes} - "
                          f"Reward: {episode_reward:.2f} - "
                          f"Average Reward (100 ep): {avg_reward:.2f} - "
                          f"Best Reward: {best_reward:.2f} - "
                          f"Epsilon: {self.current_epsilon:.7f} - "
                          f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.7f}")

                self.update_hyperparameters()

        except KeyboardInterrupt:
            print("\nTraining interrupted. Cleaning up...")
            self.clear_memory()

        finally:
            self.clear_memory()


def main():
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    agent = DQNAgent(env)

    try:
        agent.train(start_episode=None)
    finally:
        env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()