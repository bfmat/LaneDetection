from __future__ import print_function

import os
import sys

from ..apply.get_simulation_screenshot import get_simulation_screenshot, TEMP_PATH
from ..infer.inference_wrapper_single_line import InferenceWrapperSingleLine

# Check that the number of command line arguments is correct
num_arguments = len(sys.argv)
if num_arguments != 2 and num_arguments != 3:
    print('Usage:', sys.argv[0], '<right line trained model> <LSTM trained model (optional)>')
    sys.exit()

# Get the path to the network models from the command line arguments
model_path = sys.argv[1]
lstm_model_path = sys.argv[2] if num_arguments == 3 else None

# Create the engine wrapper
inference_and_steering_wrapper = InferenceWrapperSingleLine(model_path, lstm_model_path)

# Clear old data from the temp folder and record an initial output
os.system('rm %s*sim*' % TEMP_PATH)
os.system('echo 0.0 > %s-1sim.txt' % TEMP_PATH)

# Loop forever, classifying images and recording outputs to files
i = 0
while True:
    # Get the greatest-numbered image in the temp folder
    image = get_simulation_screenshot()
    # If a valid image was not found, skip the rest of this iteration
    if image is None:
        continue

    # Calculate a steering angle with the processed image
    data = inference_and_steering_wrapper.infer(image)[0]
    # If valid data was not returned, skip the rest of this iteration
    if data is None:
        continue

    # Write the classification to a temp file and rename it
    # The steering angle is the first element of the returned collection
    os.system('echo %f > %stemp.txt' % (data[0], TEMP_PATH))
    os.system('mv %stemp.txt %s%dsim.txt' % (TEMP_PATH, TEMP_PATH, i))
