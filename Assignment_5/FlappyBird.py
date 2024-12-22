import flappy_bird_gymnasium
import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torchvision import transforms
from PIL import Image
import gc


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # Convert tensors to CPU before storing
        state = state.cpu()
        action = action.cpu()
        reward = reward.cpu()
        next_state = next_state.cpu()
        done = done.cpu()

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.cat(state), torch.cat(action),
                torch.cat(reward), torch.cat(next_state),
                torch.cat(done))

    def __len__(self):
        return len(self.buffer)


class FlappyAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.memory = ReplayBuffer(10000)  # Reduced buffer size

        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon_start = 0.5
        self.epsilon_final = 0.01
        self.epsilon_decay = 500000
        self.target_update = 1000
        self.steps_done = 0

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

        self.frame_stack = deque(maxlen=4)

    def preprocess_observation(self, observation):
        image = Image.fromarray(observation)
        processed = self.transform(image)
        return processed

    def get_state(self):
        if len(self.frame_stack) < 4:
            return None
        state = torch.cat(list(self.frame_stack)).unsqueeze(0)
        return state

    def select_action(self, state):
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() > epsilon:
            with torch.no_grad():
                state = state.to(self.device)
                action = self.policy_net(state).max(1)[1].view(1, 1)
                return action
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample and move to GPU only when needed
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            [b.to(self.device) for b in self.memory.sample(self.batch_size)]

        # Compute Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            next_state_values[done_batch] = 0.0
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss and optimize
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Clear cache periodically
        if self.steps_done % 100 == 0:
            torch.cuda.empty_cache()

        return loss.item()


def train():
    env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    agent = FlappyAgent()
    episodes = 100000

    for episode in range(episodes):
        obs, _ = env.reset()

        # Initialize frame stack
        for _ in range(4):
            agent.frame_stack.append(agent.preprocess_observation(obs))

        episode_reward = 0
        episode_loss = 0
        steps = 0

        while True:
            state = agent.get_state()
            action = agent.select_action(state)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # Process new observation
            agent.frame_stack.append(agent.preprocess_observation(obs))
            next_state = agent.get_state()

            # Store the transition
            reward_tensor = torch.tensor([reward], device=agent.device)
            done_tensor = torch.tensor([done], device=agent.device)

            agent.memory.push(state, action, reward_tensor, next_state, done_tensor)

            # Optimize model
            loss = agent.optimize_model()
            if loss is not None:
                episode_loss += loss

            episode_reward += reward
            steps += 1

            if steps % agent.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            if done:
                break

        # Print statistics
        print(f"Episode {episode + 1}")
        print(f"Steps: {steps}")
        print(f"Reward: {episode_reward}")
        print(f"Average Loss: {episode_loss / steps if steps > 0 else 0}")
        print(
            f"Epsilon: {agent.epsilon_final + (agent.epsilon_start - agent.epsilon_final) * np.exp(-1. * agent.steps_done / agent.epsilon_decay)}")

        # Memory statistics
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        print("-" * 50)

        # Explicit garbage collection
        if (episode + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Save model periodically
        if (episode + 1) % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f'flappy_dqn_episode_{episode + 1}.pth')

    env.close()


if __name__ == "__main__":
    train()