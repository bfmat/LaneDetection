from __future__ import print_function

import os
import sys

from scipy.misc import imread
from keras.models import load_model
from ..infer import SteeringEngine, SlidingWindowInferenceEngine

# A script for running sliding window inference on a set of test data
# and analyzing the geometric mean of the error and of the derivative of the error,
# using the bottom of the image instead of the top to avoid extreme fluctuation


# Check that the number of command line arguments is correct
if len(sys.argv) != 4:
    print(
        'Usage:', sys.argv[0], '<folder of images> <trained model one> <trained model two>')
    sys.exit()

# Create a steering engine
steering_engine = SteeringEngine(
    proportional_multiplier=0.0025,
    derivative_multiplier=0,
    max_distance_from_line=10,
    ideal_center_x=190,
    center_y=20,
    steering_limit=100
)

# Load the supplied Keras models
models = []
for argument in sys.argv[2:]:
    model_path = os.path.expanduser(sys.argv[1])
    model = load_model(model_path)
    models.append(model)

# Create a sliding window inference engine with the model
inference_engine = SlidingWindowInferenceEngine(
    model=model,
    slice_size=16,
    stride=4
)

# Get the folder name from the command line arguments
# and iterate over all of the files in the folder
folder_path = os.path.expanduser(sys.argv[1])
for file_name in os.listdir(folder_path):

    # Format the full path of the image
    file_path = '{}/{}'.format(folder_path, file_name)

    # Load the image from disk
    image = imread(file_path)

    # Run the sliding window inference engine on the image
    predictions = inference_engine.infer(image)

    # Get a proportional error and line slope (used as the derivative error) based on the predictions
