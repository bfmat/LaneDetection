from __future__ import print_function

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


# A wrapper for a PyTorch model that allows for training of a neural network with evolutionary algorithms
# Created by brendon-ai, January 2018


# The main class, which contains a model and provides utilities for initialization, randomization, and inference
class EvolutionaryModel:
    # The initializer that creates the model and sets the weights
    def __init__(self, weights=None):
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

    # Add Gaussian noise to the weights of each of the layers in the network and return a new model
    def with_noise(self):
        # Create a list to add the weights for each layer to
        weights_list_all_layers = []
        # For each of the layers that have trainable weights
        for layer in self.weighted_layers:
            # Convert the layer's weights to a NumPy array
            weights_numpy = layer.weight.data.numpy()
            # Add Gaussian noise to the weights
            weights_numpy += np.random.normal(loc=0, scale=0.00025, size=weights_numpy.shape)
            # Convert the NumPy array to a list and add it to the list for all of the layers
            weights_list_all_layers.append(weights_numpy.tolist())
        # Return a new model with the modified array of weights
        return EvolutionaryModel(weights_list_all_layers)

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
        print('Architecture: {}'.format(self.model))
        # Print the weights of each of the weighted layers
        for i in range(len(self.model)):
            if self.model[i] in self.weighted_layers:
                weights_list = self.model[i].weight.data.numpy().tolist()
                print('Weights of layer {}: {}'.format(i, weights_list))
