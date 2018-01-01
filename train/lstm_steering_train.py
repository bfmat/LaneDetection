from __future__ import print_function

import os
import sys

import numpy as np
from scipy.misc import imread

from ..infer.inference_wrapper_single_line import InferenceWrapperSingleLine
from ..model import lstm_steering_model
from ..train.common_train_features import train_and_save

# A script for training an LSTM network used for steering a car based on the constantly updating line of best fit
# Created by brendon-ai, November 2017

# Training hyperparameters
EPOCHS = 100

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

# Create a model and train it
model = lstm_steering_model()
train_and_save(
    model=model,
    trained_model_folder=trained_model_folder,
    x=np.array([lines_of_best_fit]),
    y=np.expand_dims(np.array([steering_angles]), 2),
    epochs=EPOCHS,
    batch_size=None,
    validation_split=0
)
