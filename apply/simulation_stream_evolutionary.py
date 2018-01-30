from __future__ import print_function

import glob
import itertools
import os
import sys
import time

from ..apply.get_simulation_screenshot import get_simulation_screenshot, TEMP_PATH
from ..infer.inference_wrapper_single_line import InferenceWrapperSingleLine
from ..model.evolutionary_model import EvolutionaryModel

# A script for driving in the simulation using a neural network trained with evolutionary algorithms for steering
# Created by brendon-ai, January 2018


# Paths to the temporary and permanent steering angle files
STEERING_FILE_PATH = TEMP_PATH + '0sim.txt'
TEMP_STEERING_FILE_PATH = TEMP_PATH + 'temp.txt'

# Path to the temporary file whose existence instructs the simulation to reset the car to its starting position
RESET_FILE_PATH = TEMP_PATH + 'reset_sim'

# The number of models with added noise that should be tested on each iteration of the outer loop
NUM_NOISE_MODELS = 7

# The number of seconds to test each noise model for
TEST_SECONDS = 30

# Check that the number of command line arguments is correct
num_arguments = len(sys.argv)
if num_arguments != 2:
    print('Usage:', sys.argv[0], '<right line trained model>')
    sys.exit()

# Delete all old images and data files from the temp folder
for file_path in glob.iglob(TEMP_PATH + '*sim*'):
    os.remove(file_path)

# Record initial output to the first steering angle file
os.system('echo 0.0 > %s-1sim.txt' % TEMP_PATH)

# Load the sliding window model using the path provided as a command line argument
sliding_window_model_path = os.path.expanduser(sys.argv[1])
# Create an inference wrapper using the model path
inference_wrapper = InferenceWrapperSingleLine(sliding_window_model_path)

# Create an initial evolutionary model with the default weights
base_model = EvolutionaryModel()

# Loop forever, counting up from 0
for i in itertools.count():
    # Print a summary of the model that is being used as a base line
    print('Summary of base model for iteration {}:'.format(i))
    base_model.print_summary()
    # Create a predefined number of copies of the base model with noise added
    noise_models = [base_model.with_noise() for _ in range(NUM_NOISE_MODELS)]
    # For each of the modified models
    for model in noise_models:
        # Set the ending time for the repeating inner loop to a predefined number of seconds from now
        end_time = time.time() + TEST_SECONDS
        # Loop as quickly as possible until the end time
        while time.time() < end_time:
            # Get the greatest-numbered image in the temp folder
            image = get_simulation_screenshot(True)
            # If a valid image was not found, skip the rest of this iteration
            if image is None:
                continue

            # Run inference using the inference wrapper, discarding the output (only the error variable is required)
            data = inference_wrapper.infer(image)
            # Compute a steering angle with the evolutionary model using the errors computed by the steering engine
            steering_angle = model(inference_wrapper.steering_engine.errors)

            # Write the output to a temp file and rename it
            with open(TEMP_STEERING_FILE_PATH, 'w') as temp_file:
                print(steering_angle, file=temp_file)
            os.rename(TEMP_STEERING_FILE_PATH, STEERING_FILE_PATH)
