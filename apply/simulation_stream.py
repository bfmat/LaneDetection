from __future__ import print_function

import os
import sys

import numpy as np
from scipy.misc import imread

from infer.inference_wrapper_single_line import InferenceWrapperSingleLine

# Path to look for images in and record classifications in
TEMP_PATH = "/tmp/"

# Check that the number of command line arguments is correct
if len(sys.argv) != 2:
    print('Usage:', sys.argv[0], '<right line trained model>')
    sys.exit()

# Get the path to the network model from the command line arguments
model_path = sys.argv[1]

# Create the engine wrapper
inference_and_steering_wrapper = InferenceWrapperSingleLine(model_path)

# Clear old data from the temp folder and record an initial output
os.system("rm %s*sim*" % TEMP_PATH)
os.system("echo 0.0 > %s-1sim.txt" % TEMP_PATH)

# Loop forever, classifying images and recording outputs to files
i = 0
while True:
    # Read from last image plus one (there should not be any gaps)
    path = "%ssim%d.png" % (TEMP_PATH, i)
    if os.path.isfile(path):
        # Read the file as a 32-bit floating point tensor
        image_raw = imread(path).astype(np.float32)

        # Rearrange and crop it into a format that the neural network should accept
        image = np.transpose(image_raw, (1, 0, 2))[:, 66:132, :]

        # Calculate a steering angle with the processed image
        steering_angle = inference_and_steering_wrapper.infer(image)
        print(steering_angle)

        # Write the classification to a temp file and rename it
        os.system("echo %f > %stemp.txt" % (steering_angle, TEMP_PATH))
        os.system("mv %stemp.txt %s%dsim.txt" % (TEMP_PATH, TEMP_PATH, i))

        # Increment the image counter
        i += 1
