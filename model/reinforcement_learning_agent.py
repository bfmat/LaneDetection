import collections
import random

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adadelta

# A system for steering based on the center line of the road using reinforcement learning, meaning the network
# gradually learns while attempting to drive the car
# Created by brendon-ai, January 2018


# The discount rate used for the reinforcement learning algorithm
GAMMA = 0.95
# The initial value exploration rate used for the reinforcement learning algorithm
EPSILON_INITIAL = 1.0
# The decay value by which the epsilon is multiplied every iteration
EPSILON_DECAY = 0.995
# The minimum value that epsilon can decay to
EPSILON_MIN = 0.01
# The maximum number of time steps that can be held in the agent's memory
MEMORY_CAPACITY = 2000


# The main reinforcement learning agent, including a neural network but handling training and other functionality
class ReinforcementSteeringAgent:

    # Initialize the agent including the model and other attributes
    def __init__(self, state_size, action_size):
        # Initialize the value of epsilon which will be changed over the life of the agent
        self.epsilon = EPSILON_INITIAL

        # Initialize the agent's memory as a double-ended queue with a predefined maximum capacity
        # It will store past time steps for training
        self.memory = collections.deque(maxlen=MEMORY_CAPACITY)

        # Set the provided state size and action size as global variables
        self.state_size = state_size
        self.action_size = action_size

        # Initialize the neural network model that trains as the agent learns
        # Use a rectified linear activation function
        activation = 'relu'
        # Create the neural network model simply using a series of dense layers
        self.model = Sequential([
            Dense(3, input_shape=(self.state_size,), activation=activation),
            Dense(self.action_size)
        ])

        # Use an Adam optimizer with the predefined learning rate
        optimizer = Adadelta()

        # Compile the model with a mean squared error loss
        self.model.compile(
            loss='mse',
            optimizer=optimizer
        )

    # Add a set of values packaged as a single time step to the memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Run a prediction on a state and return an array of predicted rewards for each possible action
    def predict(self, state):
        # Use the neural network to process the state directly
        network_output = self.model.predict(state)
        # Return the first element of the output on the first axis, effectively removing the single-element batch axis
        return network_output[0]

    # Act based on a provided state, choosing either to explore or to act based on past learning
    def act(self, state):
        # Choose randomly whether or not to act randomly, depending on the exploration rate
        if np.random.rand() <= self.epsilon:
            # Choose a random value less than the number of valid actions
            return random.randrange(self.action_size)
        # Otherwise, an action must be chosen based on the current state
        else:
            # Use the neural network to predict the reward for each of the valid actions
            reward_predictions = self.predict(state)
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
                reward_predictions = self.predict(next_state)
                # Get the maximum reward possible during the next state and multiply it by the discount hyperparameter
                discounted_maximum_future_reward = np.amax(reward_predictions) * GAMMA
                # Add the discounted future reward to the current reward to calculate the target used for training
                target = reward + discounted_maximum_future_reward

            # Make a prediction based on this state, but replace the reward for the action on this time step
            target_prediction = self.model.predict(state)
            target_prediction[0, action] = target
            # Train the model based on this modified prediction
            self.model.fit(x=state, y=target_prediction, epochs=1, verbose=0)

        # If the epsilon has not already gone as low as it is allowed to
        if self.epsilon > EPSILON_MIN:
            # Multiply it by the decay factor
            self.epsilon *= EPSILON_DECAY

    # Load weights into the neural network from a provided path
    def load_weights(self, path):
        self.model.load_weights(path)

    # Save the neural network's weights into a provided path
    def save_weights(self, path):
        self.model.save_weights(path)
