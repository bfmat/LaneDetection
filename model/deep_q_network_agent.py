import copy
import random

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adadelta

# A system for steering based on the center line of the road using a deep Q-network, meaning the network
# gradually learns while attempting to drive the car
# Created by brendon-ai, January 2018

# The discount rate used for the reinforcement learning algorithm
GAMMA = 0.95
# The initial value exploration rate used for the reinforcement learning algorithm
EPSILON_INITIAL = 1.0
# The decay value by which the epsilon is multiplied every iteration
EPSILON_DECAY = 0.998
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
            for state, action, reward, next_state, done in memory_random:
                # If the game ended after this action
                if done:
                    # The target reward is the reward that was gained from this action
                    target = reward
                # Otherwise, the future reward that would result from this action must be accounted for
                else:
                    # Predict the reward resulting from the next state
                    reward_predictions = self.predict(next_state)
                    # Get the maximum reward possible during the next state and multiply it by the discount
                    discounted_maximum_future_reward = np.amax(reward_predictions) * GAMMA
                    # Add the discounted future reward to the current reward
                    # to calculate the target used for training
                    target = reward + discounted_maximum_future_reward

                # Make a prediction based on this state, but replace the reward for the action on this time step
                target_prediction = self.model.predict(state)
                target_prediction[0, action] = target
                # Train the model based on this modified prediction, getting the most recent loss value
                loss = self.model.fit(x=state, y=target_prediction, epochs=1, verbose=0).history['loss'][0]
                # Yield the loss to the calling loop so that inference can be done between any pair of training runs
                yield loss
