import os
import sys

from scipy.misc import imread

# A script for training an LSTM network used for steering a car based on the constantly updating line of best fit
# Created by brendon-ai, November 2017

# Check that the number of command line arguments is correct
if len(sys.argv) != 4:
    print('Usage:', sys.argv[0], '<image folder> <left line trained model> <right line trained model>')
    sys.exit()

# Load the paths to the model and image provided as command line arguments
image_folder = os.path.expanduser(sys.argv[1])
model_paths = sys.argv[2:]

# Create an inference engine and steering engines using the supplied model
inference_engines, steering_engine = create_engines(sys.argv[2:])

# Load all of the images from the provided folder
for image_name in os.listdir(image_folder):
    # Load the image from disk, using its fully qualified path
    image_path = image_folder + '/' + image_name
    image = imread(image_path)
