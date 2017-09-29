from __future__ import print_function

import os
import sys
import numpy

from scipy import misc
from ..model import wide_slice_model
from ..train.common_train_features import train_and_save


# A script for training a CNN to output the lateral position of a road line in a wide image
# Created by brendon-ai, September 2017


# Gather the images and labels from a specified folder
def get_data(image_folder):

    # List of images and labels
    image_list = []
    label_list = []

    # For each image path in the provided folder
    for image_name in os.listdir(image_folder):

        # Format the fully qualified path of the image
        image_path = '{}/{}'.format(image_folder, image_name)

        # If the file has size zero, skip it
        if os.path.getsize(image_path) == 0:
            continue

        # Load the image as a 32-bit floating point NumPy array
        image = misc.imread(image_path).astype(numpy.float32)

        # Get the image's horizontal road line integer position from the name
        # Names should be in the format 'x[horizontal position]_y[vertical position]_[UUID].jpg'
        road_line_position = int(image_name.split('_')[0][1:])

        # Only add it if its dimensions are correct
        if image.shape[:2] == SLICE_DIMENSIONS:
            image_list.append(image)

            # Also add the corresponding label to the label list if the image is valid
            label_list.append(road_line_position)

    # Stack all of the images into a single NumPy array (defaults to stacking on axis 0)
    image_numpy_array = numpy.stack(image_list)

    return image_numpy_array, label_list


# Check that the number of command line arguments is correct
if len(sys.argv) != 3:
    print('Usage:', sys.argv[0], '<image folder> <trained model folder>')
    sys.exit()

# Training parameters
EPOCHS = 200
BATCH_SIZE = 5
VALIDATION_SPLIT = 0.1

# Dimensions of slices not including channels
SLICE_DIMENSIONS = (16, 320)

# Create the model with specified window size
model = wide_slice_model(SLICE_DIMENSIONS)

# Load data from the folder given as a command line argument
image_folder = os.path.expanduser(sys.argv[1])
images, labels = get_data(image_folder)

# We will save snapshots to the folder path provided as the second parameter
trained_model_folder = os.path.expanduser(sys.argv[2])

# Train the model
train_and_save(
    model=model,
    trained_model_folder=trained_model_folder,
    images=images,
    labels=labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT
)
