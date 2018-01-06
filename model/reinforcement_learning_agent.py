import collections
import random

import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# A system for steering based on the center line of the road using reinforcement learning, meaning the network
# gradually learns while attempting to drive the car
# Created by brendon-ai, January 2018


# The discount rate used for the reinforcement learning algorithm
GAMMA = 0.95
# The exploration rate used for the reinforcement learning algorithm
EPSILON = 1.0
# The decay value by which the epsilon is multiplied every iteration
EPSILON_DECAY = 0.995
# The minimum value that epsilon can decay to
EPSILON_MIN = 0.01
# The learning rate for training the network
LEARNING_RATE = 0.001
# The maximum number of time steps that can be held in the agent's memory
MEMORY_CAPACITY = 2000


# The main reinforcement learning agent, including a neural network but handling training and other functionality
class ReinforcementSteeringAgent:

    # Initialize the agent including the model and other attributes
    def __init__(self, state_size, action_size):
        # Initialize the agent's memory as a double-ended queue with a predefined maximum capacity
        # It will store past time steps for training
        self.memory = collections.deque(maxlen=MEMORY_CAPACITY)

        # Set the provided state size and action size as global variables
        self.state_size = state_size
        self.action_size = action_size

        # Initialize the neural network model that trains as the agent learns
        # Use a hyperbolic tangent activation function
        activation = 'tanh'
        # Create the neural network model simply using a series of dense layers
        self.model = Sequential([
            Dense(10, input_shape=2, activation=activation),
            Dense(4, activation=activation),
            Dense(1)
        ])
        # Compile the model with a mean squared error loss and an Adadelta optimizer
        self.model.compile(
            loss='mse',
            optimizer='adadelta'
        )

    # Add a set of values packaged as a single time step to the memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Act based on a provided state, choosing either to explore or to act based on past learning
    def act(self, state):
        # Choose randomly whether or not to act randomly, depending on the exploration rate
        if np.random.rand() <= EPSILON:
            # Choose a random value less than the number of valid actions
            return random.randrange(self.action_size)
        # Otherwise, an action must be chosen based on the current state
        else:
            # Use the neural network to predict the reward for each of the valid actions
            reward_predictions = self.model.predict(state)[0]
            # The actions is the index of the maximum predicted reward
            return np.argmax(reward_predictions)

    # Train the neural network model using the actions in the memory
    def replay(self, batch_size):
        # Randomly sample a batch of experiences from the memory
        batch = random.sample(self.memory, batch_size)
        # Extract all of the information stored in the batch of experiences
        for state, action, reward, next_state, done in batch:
            # If the game ended after this action
            if done:
                # The target reward is the reward that was gained from this action
                target = reward
            # Otherwise, the future reward that would result from this action must be accounted for
            else:
                # Predict the reward resulting from the next state
                reward_predictions = self.model.predict(state)[0]
                # Get the maximum reward possible during the next state and multiply it by the discount hyperparameter
                discounted_maximum_future_reward = np.amax(reward_predictions) * GAMMA
                # Add the discounted future reward to the current reward to calculate the target used for training
                target = reward + discounted_maximum_future_reward

            # Make a prediction based on this state, but replace the reward for the action on this time step
            target_prediction = self.model.predict(state)
            target_prediction[0][action] = target
            # Train the model based on this modified prediction
            self.model.fit(x=state, y=target_prediction, epochs=1, verbose=0)

        # If the epsilon has not already gone as low as it is allowed to
        if EPSILON > EPSILON_MIN:
            # Multiply it by the decay factor
            EPSILON *= EPSILON_DECAY

    # Load weights into the neural network from a provided path
    def load_weights(self, path):
        self.model.load_weights(path)

    # Save the neural network's weights into a provided path
    def save_weights(self, path):
        self.model.save_weights(path)
