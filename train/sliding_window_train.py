#!/usr/bin/env python3

import os
import sys
import time
import numpy
import random

from scipy import misc
from ..model.sliding_window_model import sliding_window_model


# Slice up an image into square windows given the horizontal position of the road line within it
def slice_image(full_image, road_line_position):

    # Compute a slice start and a slice end given a horizontal center point
    def compute_slice(center_position):

        # Determine the horizontal starting and ending points of the window based on the global window size
        horizontal_slice_start = center_position - (WINDOW_SIZE // 2)
        horizontal_slice_end = horizontal_slice_start + WINDOW_SIZE

        return horizontal_slice_start, horizontal_slice_end

    # Slice out the positive example in the image, centered on the road line position
    positive_slice_start, positive_slice_end = compute_slice(road_line_position)
    positive_example = full_image[:, positive_slice_start:positive_slice_end, :]

    # Pick a random position outside of the range road_line_position +/- WINDOW_SIZE
    full_image_range = set(range(320))

    # Do not center the negative example within the range of the last window
    positive_example_range = set(range(positive_slice_start, positive_slice_end))

    # Get the elements that are in full_image_range but not in positive_example_range
    negative_example_range = full_image_range.difference(positive_example_range)

    # Choose a position randomly from the range
    negative_example_position = random.choice(tuple(negative_example_range))

    # Slice out the negative example centered on the random point
    negative_slice_start, negative_slice_end = compute_slice(negative_example_position)
    negative_example = full_image[:, negative_slice_start:negative_slice_end, :]

    # Return [1, 0] for the labels because the first example is always positive and the second is always negative
    return [positive_example, negative_example], [1, 0]


# Gather the images and labels from a specified folder
def get_data(image_folder):

    # List of images and labels
    image_list = []
    label_list = []

    # For each image path in the provided folder
    for image_name in os.listdir(image_folder):

        # Format the fully qualified path of the image
        image_path = '{}/{}'.format(image_folder, image_name)

        # Load the image as a 32-bit floating point NumPy array
        full_image = misc.imread(image_path).astype(numpy.float32)

        # Get the image's horizontal road line integer position from the name
        # Names should be in the format 'x[horizontal position]_y[vertical position]_[UUID].jpg'
        road_line_position = int(image_name.split('_')[0][1:])

        # Slice up the image into square windows
        image_slices, slice_labels = slice_image(full_image, road_line_position)

        # Add each of the windows to the image list, provided their shapes are correct
        for i in range(len(image_slices)):

            # Only add it if its dimensions are correct
            if image_slices[i].shape == (WINDOW_SIZE, WINDOW_SIZE, 3):
                image_list.append(image_slices[i])

                # Also add the corresponding label to the label list if the image is valid
                label_list.append(slice_labels[i])

    # Stack all of the images into a single NumPy array (defaults to stacking on axis 0)
    image_numpy_array = numpy.stack(image_list)

    return image_numpy_array, label_list


# Check that the number of command line arguments is correct
if len(sys.argv) != 3:
    print('Usage: {} <training image folder> <trained model folder>'.format(sys.argv[0]))
    sys.exit()

# Training parameters
EPOCHS = 30
BATCH_SIZE = 5
VALIDATION_SPLIT = 0.1

# Width and height of square training images
WINDOW_SIZE = 16

# Create the model with specified window size
model = sliding_window_model(WINDOW_SIZE)

# Print a summary of the model architecture
print('\nSummary of model:')
print(model.summary())

# Load data from the folder given as a command line argument
image_folder = os.path.expanduser(sys.argv[1])
images, labels = get_data(image_folder)

# Train the model
model.fit(
    images,
    labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT
)

# We will save the model to the folder path provided as the second parameter
model_folder = os.path.expanduser(sys.argv[2])

# Name the model with the current Unix time
unix_time = int(time.time())

# Format the name and save the model
trained_model_path = '{}/{}.h5'.format(model_folder, unix_time)
model.save(trained_model_path)
