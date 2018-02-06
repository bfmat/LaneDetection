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
# Use only one for a stochastic algorithm that will be less predictable but hopefully quicker to converge
NUM_NOISE_MODELS = 7

# The number of seconds to test each noise model for
TEST_SECONDS = 60

# An arbitrary large number that is higher than any variance values that will be encountered during real use
LARGE_VARIANCE = 1000

# The initial value of the learning rate (the standard deviation of the Gaussian distribution on which noise
# to be added to a randomly chosen weight of the network is generated)
INITIAL_LEARNING_RATE = 0.002

# The number that the learning rate is multiplied by every iteration, to gradually reduce it so the network can converge
LEARNING_RATE_DECAY = 0.99

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

# Create an initial evolutionary model with the default learning rate
base_model = EvolutionaryModel(INITIAL_LEARNING_RATE)
# Initialize the variable that holds the proportional variance of the base model, starting at an arbitrary large number
base_model_proportional_variance = LARGE_VARIANCE

# Loop forever, counting up from 0
for i in itertools.count():
    # Multiply the model's learning rate by the decay constant
    base_model.learning_rate *= LEARNING_RATE_DECAY

    # Print a summary of the model that is being used as a base line
    print('Summary of base model for iteration {}:'.format(i))
    base_model.print_summary()

    # Create a predefined number of copies of the base model with noise added
    noise_models = [base_model.with_noise() for _ in range(NUM_NOISE_MODELS)]
    # Create a corresponding list of proportional variances (mean squared error)
    proportional_variances = []

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

            # Run inference using the inference wrapper, discarding the output (only the error variable is required)
            inference_wrapper.infer(image)
            # Get the errors computed by the steering engine
            errors = inference_wrapper.steering_engine.errors
            # Append the first element (the proportional error) to the list
            proportional_errors.append(errors[0])
            # Compute a steering angle with the evolutionary model using the proportional and derivative errors
            steering_angle = model(errors)

            # Write the output to a temp file and rename it
            with open(TEMP_STEERING_FILE_PATH, 'w') as temp_file:
                print(steering_angle, file=temp_file)
            os.rename(TEMP_STEERING_FILE_PATH, STEERING_FILE_PATH)

        # If the error list is not empty
        if proportional_errors:
            # Compute the mean squared proportional error
            squared_errors = [error ** 2 for error in proportional_errors]
            variance = sum(squared_errors) / len(squared_errors)
            proportional_variances.append(variance)
            # Log the variance for this model
            print('Proportional variance for model {}: {}'.format(model_index, variance))
        # Otherwise, add a large number to the list and log an error message
        else:
            proportional_variances.append(LARGE_VARIANCE)
            print('List of errors was empty for model {}'.format(model_index))

        # Create the reset file so that the car will restart at the beginning
        open(RESET_FILE_PATH, 'a').close()

    # The best tested model is the one that has the lowest average proportional error
    # Check if this lowest error is less than the original error of the base model
    min_variance = min(proportional_variances)
    if min_variance < base_model_proportional_variance:
        # Get the index of this model and make it the new base model
        best_model_index = proportional_variances.index(min(proportional_variances))
        base_model = noise_models[best_model_index]
        # Print a log that says the base model has been updated
        print('Base model updated; proportional variance decreased from {} last iteration to {} this iteration'
              .format(base_model_proportional_variance, min_variance))
        # Set the proportional variance for this model
        base_model_proportional_variance = min_variance
    # Otherwise, the best performance this iteration was worse than last iteration
    else:
        # Print a log stating that the model has not been updated
        print('Base model not updated; proportional variance last iteration was {}, minimum for this iteration is {}'
              .format(base_model_proportional_variance, min_variance))
