from __future__ import print_function

import os
import sys
import time
import numpy
import random

from scipy import misc
from keras.callbacks import ModelCheckpoint
from ..model import sliding_window_model


# Slice up an image into square windows given the horizontal position of the road line within it
def slice_image(full_image, road_line_position, is_negative_example, num_random_negative_examples):

    # Compute a slice start and a slice end given a horizontal center point
    def compute_slice(center_position):

        # Determine the horizontal starting and ending points of the window based on the global window size
        horizontal_slice_start = center_position - (WINDOW_SIZE // 2)
        horizontal_slice_end = horizontal_slice_start + WINDOW_SIZE

        return horizontal_slice_start, horizontal_slice_end

    # Slice out the true (negative if is_negative_example) example in the image, centered on the road line position
    true_slice_start, true_slice_end = compute_slice(road_line_position)
    true_example = full_image[:, true_slice_start:true_slice_end, :]

    # If the window is not a negative example, we want to pick a number of negative examples from elsewhere
    if not is_negative_example:

        # Pick a random position outside of the range road_line_position +/- WINDOW_SIZE
        full_image_range = set(range(320))

        # Do not center the negative example within the range of the last window
        true_example_range = set(range(true_slice_start, true_slice_end))

        # Get the elements that are in full_image_range but not in positive_example_range
        false_example_range = full_image_range.difference(true_example_range)

        # List for images and labels that will be returned, containing only the true example initially
        image_list = [true_example]
        label_list = [1]

        # Pick a specified number of random negative examples
        for i in range(num_random_negative_examples):

                # Choose a position randomly from the range
                false_example_position = random.choice(tuple(false_example_range))

                # Slice out the negative example centered on the random point
                false_slice_start, false_slice_end = compute_slice(false_example_position)
                false_example = full_image[:, false_slice_start:false_slice_end, :]

                # Add the image and the corresponding negative flag to the lists for return
                image_list.append(false_example)
                label_list.append(0)

        return image_list, label_list

    else:
        # Return only the true example with a negative flag
        return [true_example], [0]


# Gather the images and labels from a specified folder
def get_data(image_folders):

    # List of images and labels
    image_list = []
    label_list = []

    # For each of the two image folders
    for folder_index in range(2):

        # For each image path in the provided folder
        for image_name in os.listdir(image_folders[folder_index]):

            # Format the fully qualified path of the image
            image_path = '{}/{}'.format(image_folders[folder_index], image_name)

            # If the file has size zero, skip it
            if os.path.getsize(image_path) == 0:
                continue

            # Load the image as a 32-bit floating point NumPy array
            full_image = misc.imread(image_path).astype(numpy.float32)

            # Get the image's horizontal road line integer position from the name
            # Names should be in the format 'x[horizontal position]_y[vertical position]_[UUID].jpg'
            road_line_position = int(image_name.split('_')[0][1:])

            # Slice up the image into square windows, with the second folder used for negative examples
            image_slices, slice_labels = slice_image(
                full_image=full_image,
                road_line_position=road_line_position,
                is_negative_example=bool(folder_index),
                num_random_negative_examples=15
            )

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
if len(sys.argv) != 4:
    print('Usage:', sys.argv[0], '<positive image folder> <negative image folder> <trained model folder>')
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
image_folders = [os.path.expanduser(folder) for folder in sys.argv[1:3]]
images, labels = get_data(image_folders)

# We will save snapshots to the folder path provided as the second parameter
model_folder = os.path.expanduser(sys.argv[3])

# Name the model with the current Unix time
unix_time = int(time.time())

# Train the model and save snapshots
model.fit(
    images,
    labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[ModelCheckpoint(
        '{}/batch={}-epoch={{epoch:d}}-val_loss={{val_loss:f}}.h5'
        .format(model_folder, unix_time)
    )]
)
