from __future__ import print_function

import os
import sys

from keras.models import load_model
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QWidget, QApplication
from ..infer import SteeringEngine, SlidingWindowInferenceEngine
from ..infer.visualizer_image_processing import process_images


# Program which demonstrates the effectiveness or ineffectiveness of a lane detection model
# by displaying an image and highlighting the areas in which it predicts there are road lines
# Created by brendon-ai, September 2017


# Scaling factor for the input image, which defines the size of the window
SCALING_FACTOR = 3

# The distance from the center to the edge of the green squares placed along the road line
MARKER_RADIUS = 2

# The opacity of the heat map displayed on the two bottom images
HEAT_MAP_OPACITY = 0.7

# The ideal position for the center of the image
IDEAL_CENTER_X = 160

# Labels for each of the elements of an image data tuple
IMAGE_DATA_LABELS = ('File name', 'Predicted center', 'Error from center', 'Steering angle')


# Main PyQt5 QWidget class
class Visualizer(QWidget):

    # List of NumPy images
    image_list = None

    # List of image file names
    image_data = []

    # The image we are currently on
    image_index = 0

    # The label that displays the current image
    image_box = None

    # Call various initialization functions
    def __init__(self):

        super(Visualizer, self).__init__()

        # Check that the number of command line arguments is correct
        if len(sys.argv) != 3:
            print('Usage: {} <trained model folder> <image folder>'.format(sys.argv[0]))
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

            # Create the model and load the weights from the file (load_model does not work with lambda layers)
            model = load_model(model_path)

            # Create a sliding window inference engine with the model
            inference_engine = SlidingWindowInferenceEngine(
                model=model,
                slice_size=16,
                stride=8
            )

            # Add it to the list
            inference_engines.append(inference_engine)

        # Instantiate the steering angle generation engine
        steering_engine = SteeringEngine(
            max_average_variation=40,
            proportional_multiplier=0.1,
            derivative_multiplier=0.5,
            ideal_center_x=IDEAL_CENTER_X,
            center_y_high=20,
            center_y_low=40,
            steering_limit=100
        )

        # Load and perform inference on the images
        self.image_list, self.image_data\
            = process_images(image_folder, inference_engines, steering_engine, MARKER_RADIUS, HEAT_MAP_OPACITY)

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
        self.update_display(1)

    # Update the image in the display
    def update_display(self, num_frames):

        # Update the index of the current image by whatever number is provided
        self.image_index += num_frames

        # If it is past the last image, reset it to zero or above; if it has gone below zero, reset it to the last image
        # Calculate how far it has gone beyond the relevant limit and set it to that distance past the opposite limit
        total_frames = len(self.image_list)
        if self.image_index >= total_frames:
            delta_from_last_image = self.image_index - total_frames 
            self.image_index = delta_from_last_image
        elif self.image_index < 0:
            self.image_index = len(self.image_list) + self.image_index

        # Get the image that we should display from the list
        image = self.image_list[self.image_index]

        # Convert the NumPy array into a QImage for display
        height, width, channel = image.shape
        bytes_per_line = channel * width
        current_image_qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        current_image_qpixmap = QPixmap(current_image_qimage).scaled(self.image_box.width(), self.image_box.height())

        # Fill the image box with the picture
        self.image_box.setPixmap(current_image_qpixmap)

        # Print a blank line to separate this image from the last
        print()

        # Print some metadata about the image, with the labels provided in IMAGE_DATA_LABELS
        for name, value in zip(IMAGE_DATA_LABELS, self.image_data[self.image_index]):

            # Print the name and value in a single line
            print(name, value, sep=': ')

    # Listen for key presses and update the display
    def keyPressEvent(self, event):

        # If the key is the left arrow key
        if event.key() == Qt.Key_Right:

            # Go to the next frame
            self.update_display(1)

        # If it is the right arrow key
        elif event.key() == Qt.Key_Left:

            # Go to the previous frame
            self.update_display(-1)


# If this file is being run directly, instantiate the ManualSelection class
if __name__ == '__main__':
    app = QApplication([])
    ic = Visualizer()
    sys.exit(app.exec_())
