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
        # Create the neural network using a single fully connected layer without bias
        self.model = nn.Sequential(
            nn.Linear(2, 1, bias=False)
        )
        # Use predefined working PID parameters if None is passed
        # An empty first dimension is required
        if weights is None:
            weights = [[0.0025, 0]]
        self.model[0].weight = nn.Parameter(torch.FloatTensor(weights))

    # Add Gaussian noise to the weights of each of the layers in the network and return a new model
    def with_noise(self):
        # For each of the layers in the network model
        for layer in self.model:
            # Convert the layer's weights to a NumPy array
            weights_numpy = layer.weight.data.numpy()
            # Add Gaussian noise to the weights
            weights_numpy += np.random.normal(loc=0, scale=0.00025, size=weights_numpy.size)
            # Convert the NumPy array to a list
            weights_list = weights_numpy.tolist()
            # Return a new model with the modified array of weights
            return EvolutionaryModel(weights_list)

    # Run inference on the computer center line of the road, returning the calculated steering angle
    # The __call__ name means that the model instance can itself be called as a function
    def __call__(self, center_line):
        # Convert the array to a Variable so it can be passed to the model
        center_line_variable = Variable(torch.FloatTensor(center_line))
        # Pass the data to the neural network and get a steering angle wrapped in a FloatTensor
        steering_angle_tensor = self.model(center_line_variable)
        # Return the steering angle as a floating-point number
        return steering_angle_tensor.data.numpy().tolist()[0]
