from __future__ import print_function

import os
import sys
import time

import numpy as np
import tensorflow as tf
import torch
from keras.backend.tensorflow_backend import set_session
from scipy.misc import imread
from torch import autograd
from torch import nn, optim

from ..infer.inference_wrapper_single_line import InferenceWrapperSingleLine
from ..model import lstm_steering_model

# A script for training an LSTM network used for steering a car based on the constantly updating line of best fit
# Created by brendon-ai, January 2018

# Training hyperparameters
EPOCHS = 500
LEARNING_RATE = 1e-4

# Limit the TensorFlow backend's memory usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

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
image_names = os.listdir(image_folder)
num_images = len(image_names)
for image_index in range(num_images):
    # Get the name of the current image
    image_name = image_names[image_index]
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

    # If the current image index is a multiple of 1000, notify the user
    if image_index % 1000 == 0:
        print('Loaded image', image_index, 'of', num_images)

# Convert the lines of best fit and steering angles to NumPy arrays
lines_of_best_fit_array, steering_angles_array = [
    np.array(data_list)
    for data_list in (lines_of_best_fit, steering_angles)
]
# Add dimensions to the arrays so they are 3D arrays of the format (sequences, batches, data)
lines_of_best_fit_array = lines_of_best_fit_array[:, np.newaxis, :]
steering_angles_array = steering_angles_array[:, np.newaxis, np.newaxis]
# Convert the arrays into PyTorch tensors and then into Autograd Variables that run on the GPU
x, y = [
    autograd.Variable(torch.from_numpy(data_array).float()).cuda()
    for data_array in (lines_of_best_fit_array, steering_angles_array)
]

# Create a model and use the mean squared error loss function with the Adadelta optimizer
model = lstm_steering_model()
loss_function = nn.MSELoss()
optimizer = optim.Adadelta(model.parameters())

# Make the model train on the GPU
model.cuda()

# Get the Unix time at the beginning of training
start_time = int(round(time.time()))

# Train the network one epoch at a time
for epoch in range(EPOCHS):
    # Compute the predictions by passing the entire training sequence to the network
    predictions = model(x)
    # Compute and print the loss using the predictions and the actual steering angles
    loss = loss_function(predictions, y)
    loss_number = loss.data[0]
    print('Loss of', loss_number, 'for epoch', epoch)
    # Zero the gradients for the variables that will be updated
    optimizer.zero_grad()
    # Run backpropagation, calculating gradients for each of the trainable parameters
    loss.backward()
    # Update the parameters using the optimizer
    optimizer.step()
    # Save the current model's architecture and weights in the provided path
    trained_model_path = '{}/lstm_time={}_epoch={}_loss={}.dat' \
        .format(trained_model_folder, start_time, epoch, loss_number)
    torch.save(model, trained_model_path)
