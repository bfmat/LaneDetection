import copy
import random

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adadelta

# A system for steering based on the center line of the road using a deep Q-network, meaning the network
# gradually learns while attempting to drive the car
# Created by brendon-ai, January 2018

# The base discount rate, which should be between 0 and 1
BASE_DISCOUNT = 0.8

# The initial value exploration rate used for the reinforcement learning algorithm
EPSILON_INITIAL = 1.0
# The decay value by which the epsilon is multiplied every iteration
EPSILON_DECAY = 0.9999
# The minimum value that epsilon can decay to
EPSILON_MIN = 0.01
# The minimum number of examples in the memory before training begins
MIN_TRAINING_EXAMPLES = 100


# The deep Q-network agent, including a neural network but handling training and other functionality
class DeepQNetworkAgent:

    # Initialize the agent including the model and other attributes
    def __init__(self, state_size, action_size):
        # Initialize the value of epsilon which will be changed over the life of the agent
        self.epsilon = EPSILON_INITIAL

        # Initialize the agent's memory, which will store past time steps for training
        self.memory = []

        # Set the provided state size and action size as global variables
        self.state_size = state_size
        self.action_size = action_size

        # Use a rectified linear activation function
        activation = 'tanh'
        # Create the neural network model simply using a series of dense layers
        self.model = Sequential([
            Dense(3, input_shape=(self.state_size,), activation=activation),
            Dense(5, activation=activation),
            Dense(self.action_size)
        ])
        # Use an Adam optimizer with the predefined learning rate
        optimizer = Adadelta()
        # Compile the model with a mean squared error loss
        self.model.compile(
            loss='mse',
            optimizer=optimizer
        )

    # Add a set of values packaged as a single time step to the memory, and update rewards for previous memories
    def remember(self, state, action, reward, done):
        # Add the new value to the memory as it is (it will be updated to accommodate future rewards later)
        self.memory.append([state, action, reward, done])
        # Get the index of the most recent element in the memory
        max_memory_index = len(self.memory) - 1
        # Iterate over all indices in the array, excluding the one that was just added, in reverse
        for memory_index in reversed(range(max_memory_index)):
            # If the game ended at this example, it had no bearing on future rewards, so iteration should stop
            memory_example = self.memory[memory_index]
            if memory_example[3]:
                break

            # Get the age of this memory example (the number of examples that have been added since this one)
            age = max_memory_index - memory_index
            # Take the discount to the power of the age of this example
            # This will exponentially discount the value of the current reward for older examples in the memory
            discount = BASE_DISCOUNT ** age
            # Multiply the current reward by this discount and add it to the reward for this previous example
            memory_example[2] += reward * discount

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

    # Decay the epsilon so that actions become more frequently determined by the network rather than randomly
    def decay(self):
        # If the epsilon has not already gone as low as it is allowed to
        if self.epsilon > EPSILON_MIN:
            # Multiply it by the decay factor
            self.epsilon *= EPSILON_DECAY

    # Train the neural network model; this is to be iterated over, and yields the loss or None on each iteration
    def train(self):
        # Run an infinite loop in which the training is done
        while True:
            # Yield immediately if there is less than a specified number of training examples in the memory, so that the
            # network does not quickly overfit on a very small number of examples
            if len(self.memory) < MIN_TRAINING_EXAMPLES:
                yield None

            # Iterate over the entire memory in a random order
            memory_random = copy.copy(self.memory)
            random.shuffle(memory_random)
            for state, action, reward, _ in memory_random:
                # Make a prediction based on this state, but replace the reward for the action on this time step
                target_prediction = self.model.predict(state)
                target_prediction[0, action] = reward
                # Train the model based on this modified prediction, getting the most recent loss value
                loss = self.model.fit(x=state, y=target_prediction, epochs=1, verbose=0).history['loss'][0]
                # Yield the loss to the calling loop so that inference can be done between any pair of training runs
                yield loss
