#!/usr/bin/env python3

import os
import sys
import cv2
from PyQt5.QtWidgets import QLabel, QWidget, QApplication, QPushButton
from PyQt5.QtGui import QPixmap, QPalette, QImage, QTransform, QFont, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer


# Program which allows a user to select the position of a road line in a wide image,
# in order to create training data for a convolutional neural network.
# Created by brendon-ai, September 2017

class ManualSelection(QWidget):

    image_box = None
    current_image = None

    # Call various initialization functions
    def __init__(self):

        super(ManualSelection, self).__init__()

        # Set up the UI
        self.init_ui()

        # Load the images
        load_image_paths(sys.argv[1])

    # Initialize the user interface
    def init_ui(self):

        # Black text on a light gray background
        palette = QPalette()
        palette.setColor(QPalette.Foreground, Qt.black)
        palette.setColor(QPalette.Background, Qt.lightGray)

        # Set the size, position, title, and color scheme of the window
        self.setFixedSize(1920, 860)
        self.move(0, 100)
        self.setWindowTitle('Manual Training Data Selection')
        self.setPalette(palette)

        # Initialize the image box that holds the video frames
        self.image_box = QLabel(self)
        self.image_box.setAlignment(Qt.AlignCenter)
        self.image_box.setFixedSize(1600, 528)
        self.image_box.move(10, 10)

        # Make the window exist
        self.show()

    def mousePressEvent(self, mouse_event):
        x_position = mouse_event.x()
        save_image(x_position)

    def update_image_box(self):
        self.current_image_index += 1
        current_path = self.image_paths[self.current_image_index]
        self.current_image = cv2.imread(current_path)
        current_image_pixmap = QPixmap.fromImage(self.current_image)
        self.image_box.setPixmap(current_image_pixmap)


# List to store file paths of images in
image_paths = None


# Load paths of all of the files contained in a specific folder
def load_image_paths(folder):

    # Append the folder path to the image name for every image
    for name in os.listdir(folder):
        path = folder + "/" + name
        image_paths.append(path)


# Load an image and crop it into wide slices that can be assigned a single value for the line position
def load_image(path):
    # Check if there are images left in the static list of image slices that have not yet been returned
    if len(load_image.previously_loaded_images) > 0:
        # If so, pop the last one off and return it
        return load_image.previously_loaded_images.pop()
    else:
        # Load a new image
        pass

# When the program starts, initialize the list of image slices to empty
load_image.previously_loaded_images = []


# Save an image to disk, named with the corresponding line X position
def save_image(x_position, image):
    pass


# If this file is being run directly, instantiate the ManualSelection class
if __name__ == '__main__':
    app = QApplication([])
    ic = ManualSelection()
    sys.exit(app.exec_())
