from __future__ import print_function

import os
import sys

import numpy as np
from scipy.misc import imread

from ..infer.inference_wrapper_single_line import InferenceWrapperSingleLine

# Path to look for images in and record classifications in
TEMP_PATH = '/tmp/'

# Check that the number of command line arguments is correct
if len(sys.argv) != 2:
    print('Usage:', sys.argv[0], '<right line trained model>')
    sys.exit()

# Get the path to the network model from the command line arguments
model_path = sys.argv[1]

# Create the engine wrapper
inference_and_steering_wrapper = InferenceWrapperSingleLine(model_path)

# Clear old data from the temp folder and record an initial output
os.system('rm %s*sim*' % TEMP_PATH)
os.system('echo 0.0 > %s-1sim.txt' % TEMP_PATH)

# Loop forever, classifying images and recording outputs to files
i = 0
while True:
    # Parse the names of each of the images in the temp folder and convert them to numbers
    image_names = os.listdir(TEMP_PATH)
    image_numbers = [int(name[3:-4]) for name in image_names if 'sim' in name and '.png' in name]
    # If there are no numbered images, skip the rest of the loop
    if not image_numbers:
        continue
    # Get the maximum number and format it into a file name
    max_number = max(image_numbers)
    max_numbered_path = '{}sim{}.png'.format(TEMP_PATH, max_number)

    # If the file exists
    if os.path.isfile(max_numbered_path):
        # Read the file as a 32-bit floating point tensor
        image_raw = imread(max_numbered_path).astype(np.float32)

        # Rearrange and crop it into a format that the neural network should accept
        image = np.transpose(image_raw, (1, 0, 2))[:, 90:]

        # Calculate a steering angle with the processed image
        data = inference_and_steering_wrapper.infer(image)[0]

        # If valid data has been returned
        if data is not None:
            # Write the classification to a temp file and rename it
            # The steering angle is the first element of the returned collection
            os.system('echo %f > %stemp.txt' % (data[0], TEMP_PATH))
            os.system('mv %stemp.txt %s%dsim.txt' % (TEMP_PATH, TEMP_PATH, i))
