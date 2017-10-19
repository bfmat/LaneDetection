from __future__ import print_function

import os
import sys
import math

from scipy.misc import imread
from keras.models import load_model
from ..infer import SteeringEngine, SlidingWindowInferenceEngine


# A script for running sliding window inference on a set of test data
# and analyzing the geometric mean of the error and of the derivative of the error,
# using the bottom of the image instead of the top to avoid extreme fluctuation


# The number of error values returned by the steering engine
NUM_ERRORS = 2

error_names = ['proportional', 'derivative']

# Check that the number of command line arguments is correct
if len(sys.argv) != 4:
    print('Usage:', sys.argv[0],
          '<folder of images> <trained model one> <trained model two>')
    sys.exit()

# Create a steering engine
steering_engine = SteeringEngine(
    proportional_multiplier=0.0025,
    derivative_multiplier=0,
    max_distance_from_line=10,
    ideal_center_x=190,
    center_y=320,
    steering_limit=100
)

# Load the supplied Keras models and create an inference engine for each of them
inference_engines = []
for argument in sys.argv[2:]:
    model_path = os.path.expanduser(argument)
    model = load_model(model_path)
    inference_engine = SlidingWindowInferenceEngine(
        model=model,
        slice_size=16,
        stride=4
    )

# List of accumulators for the squared proportional error and line slope
accumulators = [0] * NUM_ERRORS

# Get the folder name from the command line arguments
# and iterate over all of the files in the folder
folder_path = os.path.expanduser(sys.argv[1])
file_names = os.listdir(folder_path)
for file_name in file_names:

    # Format the full path of the image and load it from disk
    file_path = '{}/{}'.format(folder_path, file_name)
    image = imread(file_path)

    # Run the sliding window inference engine on the image
    predictions = inference_engine.infer(image)

    # Get a proportional error and line slope (used as the derivative error) based on the predictions
    errors = steering_engine.compute_steering_angle(predictions)[1:]

    # Add the squares of each of the errors to their corresponding accumulators
    for i in range(NUM_ERRORS):
        accumulators[i] += errors[i] ** 2

    # Notify the user that we have analyzed the current image
    print('Analyzed image', file_name)

# Get the number of files in the folder, which is used to calculate the variance
num_files = len(file_names)

# Calculate and print the standard deviation with respect to both errors, one at a time
for accumulator, error_name in zip(accumulators, error_names):
    variance = accumulator / num_files
    standard_deviation = math.sqrt(variance)
    print('Standard deviation with respect to the', error_name, 'error:',
          standard_deviation)
