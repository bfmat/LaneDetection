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


# Main PyQt5 QWidget class
class Visualizer(QWidget):

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
        load_images(inference_engine, image_folder)

        # Set up the UI
        self.init_ui()

    # Initialize the user interface
    def init_ui(self):
        pass


# Load and process the image with the provided inference engine
def load_images(inference_engine, image_folder):

    # Loop over each of the images in the folder
    for image_name in os.listdir(image_folder):

        # Load the image from disk, using its fully qualified path
        image_path = image_folder + '/' + image_name
        image = misc.imread(image_path)
        print(image.shape)

        # Run inference on the image
        line_positions = inference_engine.infer(image)


# If this file is being run directly, instantiate the ManualSelection class
if __name__ == '__main__':
    app = QApplication([])
    ic = Visualizer()
    sys.exit(app.exec_())
