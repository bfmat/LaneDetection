#!/usr/bin/env python3

import os
import sys

from scipy import misc
from keras.models import load_model
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QWidget, QApplication
from infer.steering_engine import SteeringEngine
from infer.sliding_window_inference_engine import SlidingWindowInferenceEngine


# Program which demonstrates the effectiveness or ineffectiveness of a lane detection model
# by displaying an image and highlighting the areas in which it predicts there are road lines.
# Created by brendon-ai, September 2017


# Scaling factor for the input image, which defines the size of the window
SCALING_FACTOR = 8

# The distance from the center to the edge of the green squares placed along the road line
MARKER_RADIUS = 2


# Main PyQt5 QWidget class
class Visualizer(QWidget):

    # List of NumPy images
    image_lists = None

    # The image we are currently on
    image_index = 0

    # The label that displays the current image
    image_box = None

    # Call various initialization functions
    def __init__(self):

        super(Visualizer, self).__init__()

        # Check that the number of command line arguments is correct
        if len(sys.argv) != 3:
            print('Usage: {} <trained model> <image folder>'.format(sys.argv[0]))
            sys.exit()

        # Process the paths to the model and image provided as command line arguments
        model_folder = os.path.expanduser(sys.argv[1])
        image_folder = os.path.expanduser(sys.argv[2])

        # Array of inference engines
        inference_engines = []

        # Get the models from the folder
        for model_name in os.listdir(model_folder):

            # Get the fully qualified path of the model
            model_path = '{}/{}'.format(model_folder, model_name)

            # Load the model
            model = load_model(model_path)

            # Create a sliding window inference engine with the model
            inference_engine = SlidingWindowInferenceEngine(
                model=model,
                window_size=16,
                stride=8
            )

            # Add it to the list
            inference_engines.append(inference_engine)

        # Load and perform inference on the images
        self.image_list = load_images(inference_engines, image_folder)

        # Set the global image height and width variables
        image_height, image_width = self.image_list[0].shape[:2]

        # Set up the UI
        self.init_ui(image_height, image_width)

    # Initialize the user interface
    def init_ui(self, image_height, image_width):

        # The window's size is that of the image times SCALING_FACTOR
        window_width = image_width * SCALING_FACTOR
        window_height = image_height * SCALING_FACTOR

        # Set the size, position, title, and color scheme of the window
        self.setFixedSize(window_width, window_height)
        self.move(100, 100)
        self.setWindowTitle('Manual Training Data Selection')

        # Initialize the image box that holds the video frames
        self.image_box = QLabel(self)
        self.image_box.setAlignment(Qt.AlignCenter)
        self.image_box.setFixedSize(window_width, window_height)
        self.image_box.move(0, 0)

        # Make the window exist
        self.show()

        # Display the initial image
        self.update_display()

    # Update the image in the display
    def update_display(self):

        # Get the image that we should display from the list
        image = self.image_list[self.image_index]

        # Convert the NumPy array into a QImage for display
        height, width, channel = image.shape
        bytes_per_line = channel * width
        current_image_qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        current_image_qpixmap = QPixmap(current_image_qimage).scaled(self.image_box.width(), self.image_box.height())

        # Fill the image box with the picture
        self.image_box.setPixmap(current_image_qpixmap)

        # Update the index of the current image
        self.image_index += 1

        # If it has passed the last image, reset it to 0
        if self.image_index == len(self.image_list):
            self.image_index = 0

        # Update again in 30 milliseconds
        QTimer().singleShot(30, self.update_display)


# Load and process the image with the provided inference engine
def load_images(inference_engines, image_folder):

    # Notify the user that we have started loading the images
    print('Loading images...')

    # Instantiate the steering angle generation engine
    steering_engine = SteeringEngine(
        max_line_variation=25,
        steering_multiplier=0.1,
        ideal_center_x=160,
        center_point_height=20
    )

    # Calculate the relative horizontal and vertical range of the position markers
    marker_range = range(-MARKER_RADIUS, MARKER_RADIUS)

    # List that we will add processed images to
    image_list = []

    # Loop over each of the images in the folder
    for image_name in sorted(os.listdir(image_folder)):

        # Load the image from disk, using its fully qualified path
        image_path = image_folder + '/' + image_name
        image = misc.imread(image_path)

        # List of points on the lines
        line_positions = []

        # With each of the provided engines
        for inference_engine in inference_engines:

            # Perform inference on the current image, adding the results to the list of points
            line_positions += inference_engine.infer(image)

        # Calculate a steering angle from the points
        steering_angle = steering_engine.compute_steering_angle(line_positions, line_positions)

        print(steering_angle)

        # For each of the positions which include horizontal and vertical values
        for position in line_positions:

            # Create a green square centered at position
            # Iterate over both dimensions
            for i in marker_range:
                for j in marker_range:

                    # Set the current pixel to green
                    image[position[1] + i, position[0] + j] = (0, 0, 0)

        # Add the prepared image to the list
        image_list.append(image)

    return image_list


# If this file is being run directly, instantiate the ManualSelection class
if __name__ == '__main__':
    app = QApplication([])
    ic = Visualizer()
    sys.exit(app.exec_())
