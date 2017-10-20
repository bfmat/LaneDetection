from __future__ import print_function

import os
import sys
import numpy

from keras.models import load_model
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QPalette, QFont
from PyQt5.QtWidgets import QLabel, QWidget, QApplication

from ..infer import SteeringEngine, SlidingWindowInferenceEngine
from ..visualizer.visualizer_image_processing import process_images


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
IDEAL_CENTER_X = 190

# Height of the line graph section of the UI
LINE_GRAPH_HEIGHT = 300

# Height of the border section above and below the guide lines on the line graph
LINE_GRAPH_BORDER_HEIGHT = 20

# Width and height of the labels on the horizontal edge of the bar graph
LINE_GRAPH_LABEL_SIZE = 40

# The absolute value of the steering angle at which the positive and negative line graph guide lines are drawn
LINE_GRAPH_GUIDE_LINE_STEERING_ANGLE = 0.1

# Labels for each of the elements of an image data tuple
IMAGE_DATA_LABELS = ('File name', 'Steering angle')

# The number of lists of points that will be drawn on the line graph
POINT_LIST_COUNT = 1


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

    # List of lists of points to be drawn on the line graph
    line_point_lists = [[]] * POINT_LIST_COUNT

    # List of steering angles corresponding to the images
    steering_angles = []

    # Vertical center of the line graph
    line_graph_center = None

    # Multiplier to convert a steering angle into pixels from the vertical center of the line graph
    line_graph_multiplier = None

    # The right bound of the line graph
    line_graph_right_bound = None

    # Call various initialization functions
    def __init__(self):

        # Call the superclass initializer
        super(Visualizer, self).__init__()

        # Check that the number of command line arguments is correct
        if len(sys.argv) != 3:
            print('Usage: {} <trained model folder> <image folder>'.format(
                sys.argv[0]))
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
                slice_size=16,
                stride=4
            )

            # Add it to the list
            inference_engines.append(inference_engine)

        # Instantiate the steering angle generation engine
        steering_engine = SteeringEngine(
            proportional_multiplier=0.0025,
            derivative_multiplier=0,
            max_distance_from_line=10,
            ideal_center_x=IDEAL_CENTER_X,
            center_y=20,
            steering_limit=100
        )

        # Load and perform inference on the images
        self.image_list, self.steering_angles, self.image_data\
            = process_images(image_folder, inference_engines, steering_engine, MARKER_RADIUS, HEAT_MAP_OPACITY)

        # Set the global image height and width variables
        image_height, image_width = self.image_list[0].shape[:2]

        # Calculate the height of one vertical half of the line graph ignoring the border
        half_graph_height_minus_border = (
            LINE_GRAPH_HEIGHT / 2) - LINE_GRAPH_BORDER_HEIGHT

        # Use that, divided by the predefined guide line steering angle, to calculate the line graph multiplier
        self.line_graph_multiplier = int(
            half_graph_height_minus_border / LINE_GRAPH_GUIDE_LINE_STEERING_ANGLE)

        # Set up the UI
        self.init_ui(image_height, image_width)

    # Initialize the user interface
    def init_ui(self, image_height, image_width):

        # The window's size is that of the image times SCALING_FACTOR
        window_width = image_width * SCALING_FACTOR
        window_height = image_height * SCALING_FACTOR

        # Calculate the center of the line graph using the height of the image as an upper limit of the graph
        self.line_graph_center = window_height + (LINE_GRAPH_HEIGHT // 2)

        # Set the size, position, title, and color scheme of the window
        # Use the image box size plus a predefined height that will be occupied by the line graph
        self.setFixedSize(window_width, window_height + LINE_GRAPH_HEIGHT)
        self.move(100, 100)
        self.setWindowTitle('Manual Training Data Selection')

        # Calculate the right bound of the line graph, by offsetting it a certain amount from the right edge
        self.line_graph_right_bound = self.width() - LINE_GRAPH_LABEL_SIZE

        # Use white text on a dark gray background
        palette = QPalette()
        palette.setColor(QPalette.Foreground, Qt.black)
        palette.setColor(QPalette.Background, Qt.lightGray)
        self.setPalette(palette)

        # Initialize the image box that holds the video frames
        self.image_box = QLabel(self)
        self.image_box.setAlignment(Qt.AlignCenter)
        self.image_box.setFixedSize(window_width, window_height)
        self.image_box.move(0, 0)

        # Font to use for the labels
        font = QFont('Source Sans Pro')
        font.setPointSize(12)

        # Create labels on the bar graph for the steering angles at which guide lines are drawn
        steering_angle_range = numpy.arange(-LINE_GRAPH_GUIDE_LINE_STEERING_ANGLE,
                                            LINE_GRAPH_GUIDE_LINE_STEERING_ANGLE, LINE_GRAPH_GUIDE_LINE_STEERING_ANGLE)
        for steering_angle in steering_angle_range:
            y_position = self.get_line_graph_y_position(steering_angle)
            line_graph_label = QLabel(self)
            line_graph_label.setFont(font)
            line_graph_label.move(self.line_graph_right_bound, y_position)
            line_graph_label.setFixedSize(
                LINE_GRAPH_LABEL_SIZE, LINE_GRAPH_LABEL_SIZE)
            line_graph_label.setText(str(steering_angle))

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
        current_image_qimage = QImage(
            image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        current_image_qpixmap = QPixmap(current_image_qimage).scaled(
            self.image_box.width(), self.image_box.height())

        # Fill the image box with the picture
        self.image_box.setPixmap(current_image_qpixmap)

        # Add a new point to the line graph five pixels left of the right edge
        y_point = self.get_line_graph_y_position(
            self.steering_angles[self.image_index])
        self.line_point_lists[0].append([self.line_graph_right_bound, y_point])

        # Shift all the points on the graph left by 5 pixels
        for point in self.line_point_lists[0]:
            point[0] -= 5

        # Force PyQt to repaint all of the lines
        self.repaint()

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

    # Called when it is time to redraw
    def paintEvent(self, _):

        # Initialize the drawing tool
        painter = QPainter(self)

        # Draw a jagged line over a list of points
        def paint_line(point_list, color):

            # Configure the line color and width
            pen = QPen()
            pen.setColor(color)
            pen.setWidth(3)
            painter.setPen(pen)

            # Iterate over the points and draw a line between each consecutive pair
            for i in range(1, len(point_list)):
                previous_point = point_list[i - 1]
                current_point = point_list[i]
                line_parameters = current_point + previous_point
                painter.drawLine(*line_parameters)

        # Calculate the Y points on the graph for steering angles of -0.1, 0.0, and 0.1 respectively
        y_negative = self.get_line_graph_y_position(
            -LINE_GRAPH_GUIDE_LINE_STEERING_ANGLE)
        y_zero = self.get_line_graph_y_position(0)
        y_positive = self.get_line_graph_y_position(
            LINE_GRAPH_GUIDE_LINE_STEERING_ANGLE)

        # Draw the three grid lines
        start_x_position = 0
        end_x_position = self.line_graph_right_bound
        paint_line([[start_x_position, y_negative], [
                   end_x_position, y_negative]], QColor(0, 0, 0))
        paint_line([[start_x_position, y_zero], [
                   end_x_position, y_zero]], QColor(0, 0, 0))
        paint_line([[start_x_position, y_positive], [
                   end_x_position, y_positive]], QColor(0, 0, 0))

        # Draw each of the lists of points on the graph
        for point_list in self.line_point_lists:
            paint_line(point_list, QColor(0, 0, 0))

    # Take an arbitrary steering angle, return the Y position that angle would correspond to on the graph
    def get_line_graph_y_position(self, steering_angle):
        y_point = -int(steering_angle *
                       self.line_graph_multiplier) + self.line_graph_center
        return y_point


# If this file is being run directly, instantiate the ManualSelection class
if __name__ == '__main__':
    app = QApplication([])
    ic = Visualizer()
    sys.exit(app.exec_())
