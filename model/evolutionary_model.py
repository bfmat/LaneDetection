from __future__ import print_function

import random

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# A wrapper for a PyTorch model that allows for training of a neural network with evolutionary algorithms
# Created by brendon-ai, January 2018


# The initial value of the learning rate (the standard deviation of the Gaussian distribution on which noise
# to be added to a randomly chosen weight of the network is generated)
INITIAL_LEARNING_RATE = 0.002

# The number that the learning rate is multiplied by every iteration, to gradually reduce it so the network can converge
LEARNING_RATE_DECAY = 0.99


# The main class, which contains a model and provides utilities for initialization, randomization, and inference
class EvolutionaryModel:

    # The initializer that creates the model and sets the weights
    def __init__(self, weights=None, weight_positions=None, learning_rate=INITIAL_LEARNING_RATE):
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

        # Set the global learning rate value
        self.learning_rate = learning_rate

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

        # Choose a random point from the list of positions of weights in the network
        layer_index, row_index, column_index = random.choice(self.weight_positions)
        # Add Gaussian noise to the weight in the list of all weights at the chosen position
        noise = np.random.normal(loc=0, scale=0.00025)
        weights_list_all_layers[layer_index][row_index][column_index] += noise

        # Return a new model with the modified array of weights, the global array of weight positions,
        # and a copy of the learning rate multiplied by the decay constant
        return EvolutionaryModel(
            weights=weights_list_all_layers,
            weight_positions=self.weight_positions,
            learning_rate=self.learning_rate * LEARNING_RATE_DECAY
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
        # Print the current learning rate
        print('Current learning rate:', self.learning_rate)
        # Print the weights of each of the weighted layers
        for i in range(len(self.model)):
            if self.model[i] in self.weighted_layers:
                weights_list = self.model[i].weight.data.numpy().tolist()
                print('Weights of layer {}: {}'.format(i, weights_list))
