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


# Number of images to process per notification provided to the user
NUM_IMAGES_TO_NOTIFY_USER = 10

# List of descriptions output before their corresponding error values
ERROR_DESCRIPTIONS = [
    'Standard deviation of the position of the car with respect to the center of the road:',
    'Standard deviation of the apparent slope of the center of the road with respect to the car:'
]


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

# Get the expected number of error values
num_errors = len(ERROR_DESCRIPTIONS)

# List of accumulators for the squared proportional error and line slope
accumulators = [0] * num_errors

# Get the folder name from the command line arguments
# and iterate over all of the files in the folder
folder_path = os.path.expanduser(sys.argv[1])
file_names = os.listdir(folder_path)
num_files = len(file_names)
for i in range(num_files):

    # Get the current file name using the index
    file_name = file_names[i]

    # Format the full path of the image and load it from disk
    file_path = '{}/{}'.format(folder_path, file_name)
    image = imread(file_path)

    # Run the sliding window inference engine on the image
    predictions = inference_engine.infer(image)

    # Get a proportional error and line slope (used as the derivative error) using the predictions
    errors = steering_engine.compute_steering_angle(predictions)[1:]

    # Add the squares of each of the errors to their corresponding accumulators
    for j in range(num_errors):
        accumulators[j] += errors[j] ** 2

    # Notify the user every predefined number of images how many we have loaded
    if i % NUM_IMAGES_TO_NOTIFY_USER == 0:
        print('Loaded image', i, 'of', num_files)

# Calculate and print the standard deviation with respect to both errors, one at a time
for accumulator, error_description in zip(accumulators, ERROR_DESCRIPTIONS):
    variance = accumulator / num_files
    standard_deviation = math.sqrt(variance)
    print(error_description, standard_deviation)
