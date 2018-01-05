import os

import torch
from torch import autograd

from ..infer.line_of_best_fit import line_of_best_fit


# A system for autonomous steering using an LSTM network to generate a steering angle based on the slope and intercept
# of the line of best fit calculated for the center line of the road
# Created by brendon-ai, November 2017


# Main class, instantiated with a trained model
class LSTMSteeringEngine:
    # Trained model used for inference of steering angles
    trained_model = None
    # Externally accessible storage for the center line
    center_line_of_best_fit = None

    # Set global trained model provided as an argument
    def __init__(self, trained_model_path):
        # Get the global path of the model, not relative to the home folder
        global_path = os.path.expanduser(trained_model_path)
        # Load the model from disk and remap it onto the CPU
        self.trained_model = torch.load(global_path, map_location=lambda storage, location: storage)

    # Compute a steering angle, given points down the center of the road
    def compute_steering_angle(self, center_points):
        # Calculate the line of best fit of the provided points
        self.center_line_of_best_fit = line_of_best_fit(center_points)
        # Add two empty axes to the returned list to represent the batch and sequence
        center_line_empty_axes = [[self.center_line_of_best_fit]]
        # Convert the list to an Autograd Variable
        center_line_variable = autograd.Variable(torch.FloatTensor(center_line_empty_axes))
        # Run inference using the provided model to get a steering angle
        steering_angle = self.trained_model(center_line_variable)
        # Return the steering angle along with dummy values for the proportional error and line slope
        return steering_angle, 0, 0
