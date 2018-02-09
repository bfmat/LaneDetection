from __future__ import print_function

import copy

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# A wrapper for a PyTorch model that allows for training of a neural network with evolutionary algorithms
# Created by brendon-ai, January 2018


# The initial value of the learning rate for each neuron (the standard deviation of the Gaussian distribution on which
# noise to be added to the corresponding weight of the network is generated)
INITIAL_LEARNING_RATE = 0.002

# The number that the learning rate is multiplied by every iteration, to gradually reduce it so the network can converge
LEARNING_RATE_DECAY = 0.99

# The number of iterations that must go by without improvement before the weight to optimize is changed
ITERATIONS_BEFORE_TRAIN_WEIGHT_CHANGE = 5


# The main class, which contains a model and provides utilities for initialization, randomization, and inference
class EvolutionaryModel:

    # The initializer that creates the model and sets the weights
    def __init__(self, weights=None, weight_positions=None, train_weight_index=0, learning_rates=None):
        # Create two fully connected layers without bias and save them in a list of layers with weights
        self.weighted_layers = [
            nn.Linear(2, 2, bias=False),
            nn.Linear(2, 1, bias=False)
        ]
        # Create the neural network using the fully connected layers, with a tanh activation in the middle
        self.model = nn.Sequential(
            self.weighted_layers[0],
            nn.Tanh(),
            self.weighted_layers[1]
        )

        # Use predefined working PID parameters if None is passed
        if weights is None:
            weights = [[[-0.0025, 0], [0, 0]], [[1, 1]]]
        # Iterate over the trainable layers and zip the layer with the first dimension in the list of weights
        for layer, layer_weights in zip(self.weighted_layers, weights):
            layer.weight = nn.Parameter(torch.FloatTensor(layer_weights))

        # If a list of weight positions has not been provided
        if weight_positions is None:
            # Compose a list containing the positions of all weights in the jagged array of all weights in the network
            self.weight_positions = []
            # For each of the layers that have trainable weights
            for layer_index in range(len(self.weighted_layers)):
                # Iterate over both dimensions of the layer's weights
                layer_weights = self.weighted_layers[layer_index].weight.data.numpy().tolist()
                for row_index in range(len(layer_weights)):
                    for column_index in range(len(layer_weights[row_index])):
                        # Save the current position in the tensor of weights to the list
                        self.weight_positions.append((layer_index, row_index, column_index))
        # Otherwise, set the global list of weight positions to the provided list
        else:
            self.weight_positions = weight_positions

        # Initialize the global index in the weight positions list which represents the weight that will be trained
        # 0 is the default if no index is provided (it is not being initialized in the noise function)
        self.train_weight_index = train_weight_index

        # If no set of learning rates is provided
        if learning_rates is None:
            # Copy the list of weights to create a list of learning rates, one for each weight,
            # and set each element to the initial learning rate
            self.learning_rates = copy.deepcopy(weights)
            for layer_index, row_index, column_index in self.weight_positions:
                self.learning_rates[layer_index][row_index][column_index] = INITIAL_LEARNING_RATE
        # Otherwise, a list of learning rates has been provided
        else:
            # Just use the supplied list
            self.learning_rates = learning_rates

        # Set the global number of iterations that have gone by since the model last improved
        # It defaults to 0 and is not passed on to new models since they will only be used if the model improves
        self.iterations_since_improvement = 0

    # Called every iteration to update the learning rate and check if the weight to optimize should be changed
    def update_learning_rate(self):
        # If the number of iterations without improvement have exceeded the threshold
        if self.iterations_since_improvement > ITERATIONS_BEFORE_TRAIN_WEIGHT_CHANGE:
            # Increment the weight index, wrapping around to zero
            self.train_weight_index = (self.train_weight_index + 1) % len(self.weight_positions)
            # Reset the counter to zero
            self.iterations_since_improvement = 0
        # Otherwise, continue to attempt to optimize this weight
        else:
            # Decay the learning rate for this neuron
            layer_index, row_index, column_index = self.weight_positions[self.train_weight_index]
            self.learning_rates[layer_index][row_index][column_index] *= LEARNING_RATE_DECAY
            # Increment the counter, since one more iteration has gone by with this model
            self.iterations_since_improvement += 1

    # Add Gaussian noise to one randomly chosen weight in the network and return a new model
    def with_noise(self):
        # Create a list to add the weights for each layer to
        weights_list_all_layers = []
        # For each of the layers that have trainable weights
        for layer in self.weighted_layers:
            # Convert the layer's weights to a list
            weights_list = layer.weight.data.numpy().tolist()
            # Add it to the list for all layers
            weights_list_all_layers.append(weights_list)

        # Get the weight position corresponding to the index of the weight to train
        layer_index, row_index, column_index = self.weight_positions[self.train_weight_index]
        # Get the learning rate corresponding to the weight to train
        learning_rate = self.learning_rates[layer_index][row_index][column_index]
        # Add Gaussian noise to the weight in the list of all weights at that position
        noise = np.random.normal(loc=0, scale=learning_rate)
        weights_list_all_layers[layer_index][row_index][column_index] += noise

        # Return a new model with the modified array of weights, the global array of weight positions, the index of the
        # weight to train, and the same list of learning rates as the present model
        return EvolutionaryModel(
            weights=weights_list_all_layers,
            weight_positions=self.weight_positions,
            train_weight_index=self.train_weight_index,
            learning_rates=self.learning_rates
        )

    # Run inference on the computer center line of the road, returning the calculated steering angle
    # The __call__ name means that the model instance can itself be called as a function
    def __call__(self, center_line):
        # Convert the array to a Variable so it can be passed to the model
        center_line_variable = Variable(torch.FloatTensor(center_line))
        # Pass the data to the neural network and get a steering angle wrapped in a FloatTensor
        steering_angle_tensor = self.model(center_line_variable)
        # Return the steering angle as a floating-point number
        return steering_angle_tensor.data.numpy().tolist()[0]

    # Print a summary of the model
    def print_summary(self):
        # Print the architecture of the neural network
        print('Architecture:', self.model)
        # Print the weights and learning rates for each of the weighted layers
        for i in range(len(self.weighted_layers)):
            weights_list = self.weighted_layers[i].weight.data.numpy().tolist()
            print('Weighted layer:', i)
            print('Weights:', weights_list)
            print('Learning rates:', self.learning_rates[i])
            print('Position of selected neuron for training:', self.weight_positions[self.train_weight_index])
