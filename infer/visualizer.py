#!/usr/bin/env python3

import os
import sys

from scipy import misc
from keras.models import load_model
from PyQt5.QtWidgets import QLabel, QWidget, QApplication
from infer.sliding_window_inference_engine import SlidingWindowInferenceEngine


# Program which demonstrates the effectiveness or ineffectiveness of a lane detection model
# by displaying an image and highlighting the areas in which it predicts there are road lines.
# Created by brendon-ai, September 2017


# Scaling factor for the input image, which defines the size of the window
SCALING_FACTOR = 4

# The distance from the center to the edge of the green squares placed along the road line
MARKER_RADIUS = 2


# Main PyQt5 QWidget class
class Visualizer(QWidget):

    # List of NumPy images
    image_list = None

    # Call various initialization functions
    def __init__(self):

        super(Visualizer, self).__init__()

        # Check that the number of command line arguments is correct
        if len(sys.argv) != 3:
            print('Usage: {} <trained model> <image folder>'.format(sys.argv[0]))
            sys.exit()

        # Process the paths to the model and image provided as command line arguments
        model_path = os.path.expanduser(sys.argv[1])
        image_folder = os.path.expanduser(sys.argv[2])

        # Load the model
        model = load_model(model_path)

        # Create a sliding window inference engine with the model
        inference_engine = SlidingWindowInferenceEngine(
            model=model,
            window_size=16,
            stride=8
        )

        # Load and perform inference on the image
        self.image_list = load_images(inference_engine, image_folder)

        # Set the global image height and width variables
        image_height, image_width = self.image_list[0].shape[:2]

        # Set up the UI
        self.init_ui(image_height, image_width)

    # Initialize the user interface
    def init_ui(self, image_height, image_width):
        pass


# Load and process the image with the provided inference engine
def load_images(inference_engine, image_folder):

    # Calculate the relative horizontal and vertical range of the position markers
    marker_range = range(-MARKER_RADIUS, MARKER_RADIUS)

    # List that we will add processed images to
    image_list = []

    # Loop over each of the images in the folder
    for image_name in os.listdir(image_folder):

        # Load the image from disk, using its fully qualified path
        image_path = image_folder + '/' + image_name
        image = misc.imread(image_path)

        # Run inference on the image
        line_positions = inference_engine.infer(image)

        # For each of the positions which include horizontal and vertical values
        for position in line_positions:

            # Create a green square centered at position
            # Iterate over both dimensions
            for i in marker_range:
                for j in marker_range:

                    # Set the current pixel to green
                    image[position[0] + i, position[1] + j] = (0, 1, 0)

        # Add the prepared image to the list
        image_list.append(image)

    return image_list


# If this file is being run directly, instantiate the ManualSelection class
if __name__ == '__main__':
    app = QApplication([])
    ic = Visualizer()
    sys.exit(app.exec_())
