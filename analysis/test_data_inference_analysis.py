from __future__ import print_function

import os
import sys

from keras.models import load_model
from ..infer import SlidingWindowInferenceEngine


# A script for running sliding window inference on a set of test data
# and analyzing the geometric mean of the error and of the derivative of the error,
# using the bottom of the image instead of the top to avoid extreme fluctuation


# Check that the number of command line arguments is correct
if len(sys.argv) != 4:
    print('Usage:', sys.argv[0], '<trained model> <folder of images>')
    sys.exit()

# Load the supplied Keras model
model_path = os.path.expanduser(sys.argv[1])
model = load_model(model_path)

# Get the folder name from the command line arguments
# and iterate over all of the files in the folder
folder_path = os.path.expanduser(sys.argv[2])
for file_name in os.listdir(folder_path):

    # Format the full path of the image
    file_path = '{}/{}'.format(folder_path, file_name)
