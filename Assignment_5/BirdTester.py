import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import argparse
from collections import deque


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
        self.frames.append(frame)

    def get_state(self):
        while len(self.frames) < self.size:
            if len(self.frames) > 0:
                self.frames.append(self.frames[-1])
            else:
                zero_frame = torch.zeros(1, 1, 84, 84)
                self.frames.append(zero_frame)
        return torch.cat(list(self.frames), dim=1)


def preprocess_image(image):
    pil_image = Image.fromarray(image)
    width, height = pil_image.size
    crop_height = int(height * 0.8)
    cropped_image = pil_image.crop((0, 0, width, crop_height))
    grayscale_image = cropped_image.convert("L")
    resized_image = grayscale_image.resize((84, 84))
    preprocessed_image = np.array(resized_image) / 255.0
    tensor_image = torch.FloatTensor(preprocessed_image).unsqueeze(0).unsqueeze(0)
    return tensor_image


def visualize_agent(model_path, delay=20):
    # Initialize environment and device
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DQN(input_channels=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize frame stack
    frame_stack = FrameStack(size=4)

    while True:
        state, _ = env.reset()
        frame = preprocess_image(env.render())

        # Initialize frame stack
        frame_stack = FrameStack(size=4)
        for _ in range(4):
            frame_stack.push(frame)

        done = False
        score = 0

        while not done:
            # Get current state
            state = frame_stack.get_state()

            # Select action
            with torch.no_grad():
                state = state.to(device)
                q_values = model(state)
                action = q_values.max(1)[1].item()

            # Take action
            next_state, reward, done, truncated, info = env.step(action)

            # Update score if we passed a pipe
            if reward == 1:
                score += 1

            # Render frame
            render_frame = env.render()

            # Add score to frame
            frame_with_score = render_frame.copy()
            cv2.putText(
                frame_with_score,
                f'Score: {score}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Show frame
            cv2.imshow('Flappy Bird AI', frame_with_score)

            # Process next frame
            next_frame = preprocess_image(render_frame)
            frame_stack.push(next_frame)

            # Break if 'q' is pressed
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                env.close()
                return

            if done:
                cv2.waitKey(2000)  # Wait 2 seconds before next game


if __name__ == "__main__":

    visualize_agent('models/dqn_episode_7600.pth', 1)