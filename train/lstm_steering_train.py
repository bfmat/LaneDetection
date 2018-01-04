from __future__ import print_function

import os
import sys

import numpy as np
import torch
from scipy.misc import imread
from torch import autograd
from torch import nn, optim

from ..infer.inference_wrapper_single_line import InferenceWrapperSingleLine
from ..model import lstm_steering_model

# A script for training an LSTM network used for steering a car based on the constantly updating line of best fit
# Created by brendon-ai, January 2018

# Training hyperparameters
EPOCHS = 100
LEARNING_RATE = 1e-4

# Check that the number of command line arguments is correct
if len(sys.argv) != 4:
    print('Usage:', sys.argv[0], '<image folder> <right line trained model> <trained LSTM folder>')
    sys.exit()

# Load the paths to the image folder, sliding window models, and trained model folder provided as command line arguments
image_folder = os.path.expanduser(sys.argv[1])
sliding_window_model_path = os.path.expanduser(sys.argv[2])
trained_model_folder = os.path.expanduser(sys.argv[3])

# Create an inference and steering wrapper using the supplied model paths
inference_and_steering_wrapper = InferenceWrapperSingleLine(sliding_window_model_path)

# Create a list of steering angles and a list of lines of best fit
steering_angles = []
lines_of_best_fit = []

# Load all of the images from the provided folder
for image_name in os.listdir(image_folder):
    # Notify the user the image is being processed
    print('Loading image', image_name)

    # Load the image from disk, using its fully qualified path
    image_path = image_folder + '/' + image_name
    image = imread(image_path)

    # Run inference on the image and collect the line of best fit and steering angle
    output_values, _, _, line_of_best_fit = inference_and_steering_wrapper.infer(image)

    # If valid values were returned at all
    if output_values is not None:
        # Get the steering angle and add it to the list alongside the line of best fit
        steering_angle = output_values[0]
        steering_angles.append(steering_angle)
        lines_of_best_fit.append(line_of_best_fit)

# Convert the lines of best fit and steering angles to NumPy arrays
lines_of_best_fit_array, steering_angles_array = [
    np.array(data_list)
    for data_list in (lines_of_best_fit, steering_angles)
]
# Add dimensions to the arrays so they are 3D arrays of the format (sequences, batches, data)
lines_of_best_fit_array = lines_of_best_fit_array[:, np.newaxis, :]
steering_angles_array = steering_angles_array[:, np.newaxis, np.newaxis]
# Convert the arrays into PyTorch tensors and then into Autograd Variables
x, y = [
    autograd.Variable(torch.from_numpy(data_array).float())
    for data_array in (lines_of_best_fit_array, steering_angles_array)
]

# Create a model and use the mean squared error loss function with the Adadelta optimizer
model = lstm_steering_model()
loss_function = nn.MSELoss()
optimizer = optim.Adadelta(model.parameters())

# Train the network one epoch at a time
for epoch in range(EPOCHS):
    # Compute the predictions by passing the entire training sequence to the network
    predictions = model(x)
    # Compute and print the loss using the predictions and the actual steering angles
    loss = loss_function(predictions, y)
    print('Loss of', loss.data[0], 'for epoch', epoch)
    # Zero the gradients for the variables that will be updated
    optimizer.zero_grad()
    # Run backpropagation, calculating gradients for each of the trainable parameters
    loss.backward()
    # Update the parameters using the optimizer
    optimizer.step()
