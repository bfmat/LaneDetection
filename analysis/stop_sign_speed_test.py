from __future__ import print_function

import os
import sys
import time

from keras.models import load_model
import numpy as np
from skimage.io import imread

from ..infer.stop_sign_inference import box_stop_signs

# Tool for testing the length of time it takes to process a single image with the stop sign neural network
# Created by brendon-ai, April 2018

# Check that the number of command line arguments is correct
if len(sys.argv) != 3:
    print('Usage:', sys.argv[0], '<model path>', '<test image>')
    sys.exit()

# Load the model and the test image from disk
model_path = os.path.expanduser(sys.argv[1])
model = load_model(model_path)
image_path = os.path.expanduser(sys.argv[2])
image = imread(image_path)

# Get the time before inference
before_time = time.time()
# Run inference once on this single image, drawing on a copy of the image
box_stop_signs(model, image, np.copy(image))
# Calculate and print the amount of time it took to process the image
time_delta = time.time() - before_time
print('Time delta:', time_delta)
