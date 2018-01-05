import os

import torch

from ..infer.line_of_best_fit import line_of_best_fit


# A system for autonomous steering using an LSTM network to generate a steering angle based on the slope and intercept
# of the line of best fit calculated for the center line of the road
# Created by brendon-ai, November 2017


# Main class, instantiated with a trained model
class LSTMSteeringEngine:
    # Trained model used for inference of steering angles
    trained_model = None

    # Set global trained model provided as an argument
    def __init__(self, trained_model_path):
        # Get the global path of the model, not relative to the home folder
        global_path = os.path.expanduser(trained_model_path)
        # Load the model from disk and save it in the global variable
        self.trained_model = torch.load(global_path)

    # Compute a steering angle, given points down the center of the road
    def compute_steering_angle(self, center_points):
        # Calculate the line of best fit of the provided points
        line_parameters = line_of_best_fit(center_points)

        # Run inference using the provided model to get a steering angle
        steering_angle = self.trained_model.infer(line_parameters)

        return steering_angle
