# Deep Q-Learning Implementation for Flappy Bird
## Architecture Overview

### Neural Network Architecture
The implementation uses a Deep Q-Network (DQN) with the following structure:

1. **Convolutional Layers**:
   - Input: 4 channels (stacked frames) of 84x84 grayscale images
   - First layer: 32 filters, 8x8 kernel, stride 4, LeakyReLU(0.01)
   - Second layer: 64 filters, 4x4 kernel, stride 2, LeakyReLU(0.01)
   - Third layer: 64 filters, 3x3 kernel, stride 1, LeakyReLU(0.01)

2. **Fully Connected Layers**:
   - Flatten layer: 3136 neurons (64 * 7 * 7)
   - Hidden layer: 512 neurons with LeakyReLU(0.01)
   - Output layer: 2 neurons (representing actions)

### Key Components

1. **Frame Stack**:
   - Maintains a stack of 4 consecutive frames
   - Used to capture temporal information and motion
   - Each frame is preprocessed to 84x84 grayscale

2. **Replay Buffer**:
   - Capacity: 50,000 transitions
   - Stores (state, action, reward, next_state, done) tuples
   - Implements experience replay for stable learning

## Hyperparameters and Training Configuration

### Initial Parameters
- Learning rate: 0.00025 (Adam optimizer)
- Batch size: 32
- Gamma (discount factor): 0.99
- Initial epsilon: 0.4
- Target network update frequency: 1000 steps

### Manual Hyperparameter Tuning During Runtime
The implementation supports dynamic hyperparameter adjustments through a YAML configuration file (`hyperparameter_updates.yaml`), allowing real-time parameter tuning without stopping the training:

```yaml
status: Update/Ignore  # Controls whether updates should be applied
updates:
  epsilon:
    action: sub/add/mul  # Type of operation
    value: 0.002        # Amount to adjust
  epsilon_decay:
    action: mul
    value: 1
  learning_rate:
    action: mul/add/sub
    value: 1
```

Features:
- Parameters can be modified while the model is training
- Supports three types of operations: addition, subtraction, and multiplication
- Adjustable parameters include:
  - Epsilon (exploration rate)
  - Epsilon decay rate
  - Learning rate
- Status flag prevents repeated application of the same updates
- Changes are logged to track parameter evolution

### Automatic Epsilon Tweaking
The implementation includes an automatic epsilon adjustment mechanism that maintains the average reward between 10 and 20:
- Every 100 episodes, the system evaluates the average reward
- If average reward > 20: Increase epsilon by 0.005 (max 1.0)
- If average reward < 10: Decrease epsilon by 0.005 (min epsilon_end)
- This adaptive approach ensures optimal exploration-exploitation balance
- Code implementation:
```python
if len(rewards_history) == 100:  # Only adjust after we have 100 episodes
    if avg_reward > 20:
        self.current_epsilon = min(1.0, self.current_epsilon + 0.005)
    elif avg_reward < 10:
        self.current_epsilon = max(self.epsilon_end, self.current_epsilon - 0.005)
```

## Experimental Results and Optimization

### Initial Implementation Challenges
- High starting epsilon (>0.5) caused excessive jumps since there are a high number of frames and the actor takes a random action(0.5> change it jumps)
- Agent showed unstable behavior with frequent jumping
- Average rewards were negative (-1.75 to -0.64)
![image](https://github.com/user-attachments/assets/62b3d1c6-7521-4cc4-8f35-af3ffaf38126)

### Optimization Attempts

1. **Frame Skipping and Slower Epsilon Decay**:
   - Implemented 4-frame skipping
   - Reduced epsilon decay rate
   - Results: Improved stability and higher rewards
   - Best reward achieved: 27.70
   - Average reward increased to 9.32
![image](https://github.com/user-attachments/assets/f4a004f3-bb7b-4865-82f2-d6ede98f031b)

2. **Learning Rate and Epsilon Adjustments**:
   - Increased learning rate by 10x
   - Implemented faster epsilon decay
   - Results: Achieved best reward of 24.90
   - More consistent performance with average rewards around 10.80
![image](https://github.com/user-attachments/assets/7268925d-3bd4-4095-bbb5-190f349922e3)

3. **Architecture Modifications**:
   - Added 4-frame state input
   - Set minimum epsilon to 0.05
   - Results:
     - Best score: 81.29
     - Average score: 48.45 ± 21.82
     - Significant improvement in stability
![image](https://github.com/user-attachments/assets/58980bf8-b952-44f9-bb93-5e5eaa61617f)
![image](https://github.com/user-attachments/assets/9e5c69e1-3b1e-4f09-8c6e-c0f4b65568c7)

4. **Final Optimizations**:
   - Removed frame skipping
   - Implemented automatic epsilon tweaking
   - Target average reward: 10-20 range
   - Results: Achieved consistent high scores
     - Best score: 1328.89
     - Average score: 983.38 ± 267.32
![image](https://github.com/user-attachments/assets/bc90757e-80ab-4bc3-acc1-0feb921b7245)
![image](https://github.com/user-attachments/assets/0e508bc0-0b23-40f0-bba5-7a4a5897f3e5)

## Performance Analysis

The final implementation showed significant improvements over the initial version:
1. **Score Improvement**:
   - Initial average scores: negative values
   - Final average scores: ~983
   - Best recorded score: 1328.89

2. **Stability**:
   - Reduced variance in performance
   - More consistent learning progress
   - Better exploration-exploitation balance

3. **Learning Efficiency**:
   - Faster convergence to optimal policy
   - Better generalization across episodes
   - More robust performance across different scenarios

## Conclusions

The implementation successfully evolved through several iterations, with each modification addressing specific challenges:
1. The frame stacking mechanism proved crucial for temporal understanding
2. Automatic epsilon adjustment helped maintain optimal exploration
3. The final architecture demonstrated strong performance and stability
4. The removal of frame skipping and implementation of dynamic epsilon adjustment were key to achieving high scores
