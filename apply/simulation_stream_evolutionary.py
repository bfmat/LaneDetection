from __future__ import print_function

import glob
import itertools
import math
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
# Use only one for a stochastic algorithm that will be less predictable but hopefully quicker to converge
NUM_NOISE_MODELS = 7

# The number of seconds to test each noise model for
TEST_SECONDS = 60

# An arbitrary large number that is higher than any standard deviation values that will be encountered during real use
LARGE_STANDARD_DEVIATION = 1000

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

# Create a copy of the steering engine for error analysis purposes that will calculate errors at the bottom of the image
analysis_steering_engine = inference_wrapper.steering_engine
analysis_steering_engine.center_y = 90

# Create an initial evolutionary model with the default learning rate
base_model = EvolutionaryModel()
# Initialize the variable that holds the proportional standard deviation of the base model, starting at a large number
base_model_proportional_standard_deviation = LARGE_STANDARD_DEVIATION

# Loop forever, counting up from 0
for i in itertools.count():
    # Update the learning rate of the model
    base_model.update_learning_rate()

    # Print a summary of the model that is being used as a base line
    print('Summary of base model for iteration {}:'.format(i))
    base_model.print_summary()

    # Create a predefined number of copies of the base model with noise added
    noise_models = [base_model.with_noise() for _ in range(NUM_NOISE_MODELS)]
    # Create a corresponding list of proportional standard deviations
    proportional_standard_deviations = []

    # For each of the modified models, and a corresponding increasing index number
    for model, model_index in zip(noise_models, itertools.count()):
        # Create an empty list to which the proportional errors for this test will be appended
        proportional_errors = []
        # Set the ending time for the repeating inner loop to a predefined number of seconds from now
        end_time = time.time() + TEST_SECONDS
        # Loop as quickly as possible until the end time
        while time.time() < end_time:
            # Get the greatest-numbered image in the temp folder
            image = get_simulation_screenshot(True)
            # If a valid image was not found, skip the rest of this iteration
            if image is None:
                continue

            # Run inference using the inference wrapper and collect the center line positions
            center_line_positions = inference_wrapper.infer(image)[1]
            # Get the errors computed by the steering engine
            errors = inference_wrapper.steering_engine.errors
            # Compute a steering angle with the evolutionary model using the proportional and derivative errors
            steering_angle = model(errors)

            # Write the output to a temp file and rename it
            with open(TEMP_STEERING_FILE_PATH, 'w') as temp_file:
                print(steering_angle, file=temp_file)
            os.rename(TEMP_STEERING_FILE_PATH, STEERING_FILE_PATH)

            # Run inference with the analysis steering engine on the center line positions
            analysis_steering_engine.compute_steering_angle(center_line_positions)
            # Append the proportional error computed by the analysis engine to the list
            analysis_proportional_error = analysis_steering_engine.errors[0]
            proportional_errors.append(analysis_proportional_error)

        # If the error list is not empty
        if proportional_errors:
            # Compute the standard deviation (the square root of the mean squared proportional error)
            squared_errors = [error ** 2 for error in proportional_errors]
            variance = sum(squared_errors) / len(squared_errors)
            standard_deviation = math.sqrt(variance)
            proportional_standard_deviations.append(standard_deviation)
            # Log the standard deviation for this model
            print('Proportional standard deviation for model {}: {}'.format(model_index, standard_deviation))
        # Otherwise, add a large number to the list and log an error message
        else:
            proportional_standard_deviations.append(LARGE_STANDARD_DEVIATION)
            print('List of errors was empty for model {}'.format(model_index))

        # Create the reset file so that the car will restart at the beginning
        open(RESET_FILE_PATH, 'a').close()

    # The best tested model is the one that has the lowest proportional standard deviation
    # Check if this lowest error is less than the original error of the base model
    min_standard_deviation = min(proportional_standard_deviations)
    if min_standard_deviation < base_model_proportional_standard_deviation:
        # Get the index of this model and make it the new base model
        best_model_index = proportional_standard_deviations.index(min_standard_deviation)
        base_model = noise_models[best_model_index]
        # Print a log that says the base model has been updated
        print(
            'Base model updated; proportional standard deviation decreased from {} last iteration to {} this iteration'
                .format(base_model_proportional_standard_deviation, min_standard_deviation)
        )
        # Set the proportional standard deviation for this model
        base_model_proportional_standard_deviation = min_standard_deviation
    # Otherwise, the best performance this iteration was worse than last iteration
    else:
        # Print a log stating that the model has not been updated
        print(
            'Base model not updated; lowest standard deviation so far was {},'
                .format(base_model_proportional_standard_deviation),
            'minimum for this iteration is {}'
                .format(min_standard_deviation)
        )
