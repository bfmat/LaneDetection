#!/usr/bin/env python3

import os
import sys
import cv2
from PyQt5.QtWidgets import QLabel, QWidget, QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


# Program which allows a user to select the position of a road line in a wide image,
# in order to create training data for a convolutional neural network.
# Created by brendon-ai, September 2017

# Dimensions of window as well as image
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


# Main PyQt5 QWidget class
class ManualSelection(QWidget):

    # The label that displays the current image
    image_box = None

    # The current image as a NumPy array
    current_image = None

    # The index of the next image's path in the paths list
    current_image_index = 0

    # List of fully qualified file paths of images
    image_paths = None

    # Factor by which images are scaled up to be displayed (computed at runtime)
    image_scaling_factor = None

    # A list of two-dimensional points that the user places on the image to mark the road line
    # Used as a path for positive training examples along the line
    road_line_path = []

    # Call various initialization functions
    def __init__(self):

        super(ManualSelection, self).__init__()

        # Check that the number of command line arguments is correct
        if len(sys.argv) != 3:
            print('Usage: {} <input image folder> <output image folder>'.format(sys.argv[0]))
            sys.exit()

        # Load the images from the path supplied as a command line argument
        image_folder = os.path.expanduser(sys.argv[1])
        self.image_paths = get_image_paths(image_folder)

        # Set up the UI
        self.init_ui()

    # Initialize the user interface
    def init_ui(self):

        # Set the size, position, title, and color scheme of the window
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.move(100, 100)
        self.setWindowTitle('Manual Training Data Selection')

        # Initialize the image box that holds the video frames
        self.image_box = QLabel(self)
        self.image_box.setAlignment(Qt.AlignCenter)
        self.image_box.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.image_box.move(0, 0)

        # Make the window exist
        self.show()

        # Display the initial image
        self.update_current_image()

    def mousePressEvent(self, mouse_event):

        # Record the current mouse position to the window path list as a two-element tuple
        # Divide it by a scaling factor so that positions apply to the original unscaled image
        mouse_position = (
            mouse_event.x() // self.image_scaling_factor,
            mouse_event.y() // self.image_scaling_factor
        )
        self.road_line_path.append(mouse_position)

    def keyPressEvent(self, key_event):

        # Check if the space bar was pressed
        if key_event.key() == Qt.Key_Space:

            # Save the current image data and clear the line path
            save_training_data(self.current_image, self.road_line_path)
            self.road_line_path = []

            # If there are no images left that we haven't read, exit the application
            if self.current_image_index == len(self.image_paths):
                sys.exit()
            else:
                # If there are images left, update the current image
                self.update_current_image()

    # Update the current image, including the display box
    def update_current_image(self):

        # Get the path corresponding to the current index and load the image from disk
        current_path = self.image_paths[self.current_image_index]
        self.current_image = cv2.imread(current_path)

        # Convert the NumPy array into a QImage for display
        height, width, channel = self.current_image.shape
        bytes_per_line = channel * width
        current_image_qimage_bgr = QImage(self.current_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        current_image_qimage_rgb = current_image_qimage_bgr.rgbSwapped()
        current_image_qpixmap = QPixmap(current_image_qimage_rgb).scaled(WINDOW_WIDTH, WINDOW_HEIGHT)

        # Set the image scaling factor to the window width divided by the image width
        self.image_scaling_factor = WINDOW_WIDTH / width

        # Fill the image box with the picture
        self.image_box.setPixmap(current_image_qpixmap)

        # Update the index of the current image
        self.current_image_index += 1


# Load paths of all of the files contained in a specific folder
def get_image_paths(folder):

    # List to store file paths of images in
    image_paths = []

    # Append the folder path to the image name for every image
    for name in os.listdir(folder):
        path = folder + '/' + name
        image_paths.append(path)

    return image_paths


# Slice and save data derived from a single image
def save_training_data(image, road_line_path):
    pass


# If this file is being run directly, instantiate the ManualSelection class
if __name__ == '__main__':
    app = QApplication([])
    ic = ManualSelection()
    sys.exit(app.exec_())
