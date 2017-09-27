from __future__ import print_function

import os
import sys
import time

from scipy.misc import imread
from keras.models import load_model
from ..infer import SlidingWindowInferenceEngine


# A script for running an inference engine with a provided model and number of iterations
# and seeing how long it takes to run with a variety of configurations.
# Created by brendon-ai, September 2017


# Check that the number of command line arguments is correct
if len(sys.argv) != 4:
    print('Usage:', sys.argv[0], '<number of iterations> <trained model> <example image>')
    sys.exit()

# Parse the number of iterations
num_iterations = int(sys.argv[1])

# Load the provided model
model_path = os.path.expanduser(sys.argv[2])
model = load_model(model_path)

# Parse the example image path
image_path = os.path.expanduser(sys.argv[3])

# Create an inference engine with the model
inference_engine = SlidingWindowInferenceEngine(
    model=model,
    window_size=16,
    stride=(8, 4)
)

# Record the starting time
start_time = time.time()

# For the provided number of iterations
for i in range(num_iterations):

    # Load the image again on every iteration, for a better simulation of real-world use
    image = imread(image_path)

    # Crop the image
    image_cropped = image[:, 40:-40, :]

    # Run the inference engine on the example image, discarding the result
    inference_engine.infer(image_cropped)

    # If the current iteration is divisible by 10, tell the user how many we have completed
    if i % 10 == 0:
        print('Completed iteration', i, 'of', num_iterations)

# Record the completion time
end_time = time.time()

# Print out the elapsed time
print('Time elapsed:', end_time - start_time, 'seconds')
