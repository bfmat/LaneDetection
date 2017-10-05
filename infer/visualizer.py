from __future__ import division, print_function

import os
import sys
import numpy

from scipy.misc import imread, imsave
from keras.models import load_model
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QWidget, QApplication
from ..infer import SteeringEngine, SlidingWindowInferenceEngine
from ..infer.steering_engine import remove_outliers


# Program which demonstrates the effectiveness or ineffectiveness of a lane detection model
# by displaying an image and highlighting the areas in which it predicts there are road lines
# Created by brendon-ai, September 2017


# Scaling factor for the input image, which defines the size of the window
SCALING_FACTOR = 4

# The distance from the center to the edge of the green squares placed along the road line
MARKER_RADIUS = 2

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

        # Load and perform inference on the images
        self.image_list, self.image_data = load_images(inference_engines, image_folder)

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

        # Update the index of the current image by whatever number is provided
        self.image_index += num_frames

        # If it has passed the last image, reset it to 0
        if self.image_index == len(self.image_list):
            self.image_index = 0

    # Listen for key presses and update the display
    def keyPressEvent(self, event):

        # If the key is the left arrow key
        if event.key() == Qt.Key_Left:

            # Go to the next frame
            self.update_display(1)

        # If it is the right arrow key
        elif event.key() == Qt.Key_Right:

            # Go to the previous frame
            self.update_display(-1)


# Display a translucent red heat map over an image (modifying it in place), given a tensor of predictions to base it on
def apply_heat_map(image, prediction_tensor):

    # Find the factor to calculate rectangular blocks in the image
    # that visually correspond to the positions in the prediction tensor
    heat_map_block_shape = [image_dimension / prediction_dimension
                            for image_dimension, prediction_dimension in zip(image.shape, prediction_tensor.shape)]

    # Iterate over both dimensions of the image
    for y, x in numpy.ndindex(prediction_tensor.shape):

        # Find the bounds of the corresponding heat map box in the image and slice out the box
        block_bounds = [
            int(round((dimension + offset) * block_dimension))
            for dimension, block_dimension in zip((y, x), heat_map_block_shape)
            for offset in (0, 1)
        ]
        block = image[block_bounds[0]:block_bounds[1], block_bounds[2]:block_bounds[3]]

        # Color shift the block to red, with intensity of the red
        # proportional to the prediction corresponding to the block
        red_weight = prediction_tensor[y, x]
        red_pattern_flat = numpy.repeat([255, 0, 0], block.size / 3)
        block.flat = [
            ((red_value * red_weight) + (original_value * (1 - red_weight))) / 2
            for red_value, original_value in zip(red_pattern_flat, block.flat)
        ]


# Load and process the image with the provided inference engine
def load_images(inference_engines, image_folder):

    # Notify the user that we have started loading the images
    print('Loading images...')

    # Instantiate the steering angle generation engine
    steering_engine = SteeringEngine(
        max_average_variation=40,
        steering_multiplier=0.1,
        ideal_center_x=IDEAL_CENTER_X,
        center_y=40,
        steering_limit=0.2
    )

    # List that we will add processed images to
    image_list = []

    # List of file names of images
    image_names = sorted(os.listdir(image_folder))

    # List of image metadata for display
    image_data = []

    # Loop over each of the images in the folder
    for image_name in image_names:

        # Print out the file name of the image
        print('Loaded {}'.format(image_name))

        # Load the image from disk, using its fully qualified path
        image_path = image_folder + '/' + image_name
        image = imread(image_path)

        # List of points on the lines
        all_line_positions = []

        # With each of the provided engines
        for inference_engine in inference_engines:

            # Perform inference on the current image, adding the results to the list of points
            all_line_positions.append(inference_engine.infer(image))

        # Calculate a steering angle from the points
        values = steering_engine.compute_steering_angle(all_line_positions)

        # Set the steering angle and error to large negative values if None is returned
        if values is None:
            steering_angle = -5
            error = -5

        else:
            # Extract steering angle and error from the return values
            steering_angle, error = values

        # Remove the outliers from each of the lists and keep the outliers in a separate list
        all_line_positions_without_outliers = []
        all_line_positions_outliers_only = []
        for line_positions in all_line_positions:

            # Add the outlier-free points to one list
            line_positions_without_outliers = remove_outliers(line_positions, 40)
            all_line_positions_without_outliers.append(line_positions_without_outliers)

            # Find the points that are in the original list but not in the outlier-free list (that is, the outliers)
            all_positions_set = set(line_positions)
            outlier_free_set = set(line_positions_without_outliers)
            line_positions_outliers_only = list(all_positions_set - outlier_free_set)

            # Add it to the outlier-only list
            all_line_positions_outliers_only.append(line_positions_outliers_only)

        # Display a heat map in place on the image
        apply_heat_map(image, inference_engines[0].last_prediction_tensor)

        # Calculate the center of the road from the steering angle
        center_x = int(steering_engine.ideal_center_x - error)

        # Create a vertical blue line at the same X position as the predicted center of the road, if possible
        try:
            image[:, center_x] = [0, 0, 255]
        except:
            pass

        # Create a vertical black line at the predefined center of the image
        image[:, IDEAL_CENTER_X] = 0

        # Combine the two road lines with the lists of outliers, along with their corresponding colors
        lines_and_colors = [
            (all_line_positions_without_outliers[0], [0, 0, 255]),
            (all_line_positions_without_outliers[1], [0, 255, 0]),
            (all_line_positions_outliers_only[0], [255, 0, 0]),
            (all_line_positions_outliers_only[1], [255, 255, 0])
        ]

        # For each of the two road lines
        for line_positions, color in lines_and_colors:

            # For each of the positions which include horizontal and vertical values
            for position in line_positions:

                # Calculate the four bounds of the marker to be placed
                bounds = [int(round(center + offset)) for center in position for offset in (-MARKER_RADIUS, MARKER_RADIUS)]

                # Create a black square within the bounds
                image[bounds[2]:bounds[3], bounds[0]:bounds[1]] = color

        # Add the prepared image to the list
        image_list.append(image)

        # Add the corresponding name, center position, error, and steering angle to the data list
        image_data.append((image_name, center_x, error, steering_angle))

    # Notify the user that loading is complete
    print('Loading complete!')

    # Return the images and their corresponding names
    return image_list, image_data


# If this file is being run directly, instantiate the ManualSelection class
if __name__ == '__main__':
    app = QApplication([])
    ic = Visualizer()
    sys.exit(app.exec_())
