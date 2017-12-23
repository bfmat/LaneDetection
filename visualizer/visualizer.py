# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QPalette, QFont
from PyQt5.QtWidgets import QLabel, QWidget, QApplication

from ..infer.inference_wrapper_single_line import InferenceWrapperSingleLine
from ..infer.inference_wrapper_two_lines import InferenceWrapperTwoLines
from ..visualizer.visualizer_image_processing import process_images

# Program which demonstrates the effectiveness or ineffectiveness of a lane detection model
# by displaying an image and highlighting the areas in which it predicts there are road lines
# Created by brendon-ai, September 2017

# Font used in varying sizes throughout the user interface
UI_FONT_NAME = 'Source Sans Pro'

# Scaling factor for the input image, which defines the size of the window
SCALING_FACTOR = 3

# The distance from the center to the edge of the green squares placed along the road line
MARKER_RADIUS = 2

# The opacity of the heat map displayed on the two bottom images
HEAT_MAP_OPACITY = 0.7

# Height of the line graph section of the UI
LINE_GRAPH_HEIGHT = 300

# Height of the border section above and below the guide lines on the line graph
LINE_GRAPH_BORDER_HEIGHT = 20

# Size (width and height) and font size of the labels on the horizontal edge of the line graph
LINE_GRAPH_LABEL_SIZE = 40
LINE_GRAPH_LABELS_FONT_SIZE = 12

# Height and font size of the legend labels below the line graph
LINE_GRAPH_LEGEND_HEIGHT = 30
LINE_GRAPH_LEGEND_FONT_SIZE = 12

# The absolute value of the unscaled number at which the
# positive and negative line graph guide lines are drawn
LINE_GRAPH_GUIDE_LINE_ABSOLUTE_VALUE = 1

# The height, text contents, and font of the labels
# that identify the heat maps in the user interface
HEAT_MAP_LABELS_HEIGHT = 50
HEAT_MAP_LABELS_TEXT = ['Left line heat map', 'Right line heat map']
HEAT_MAP_LABELS_FONT_SIZE = 16

# Labels for each of the elements of an image data tuple
IMAGE_DATA_LABELS = ['File name', 'Steering angle']

# The descriptions, multipliers, and corresponding colors
# of the lines that will be drawn on the line graph
LINE_DATA = [('Steering angle', 10, 'yellow', 'tenths of motor rotations'),
             ('Proportional error', 0.01, 'cyan', 'hundreds of pixels'),
             ('Derivative error', 1, 'magenta', 'slope of line')]

# The value of the light gray filler color, from 0 to 255
LIGHT_GRAY_COLOR = 192


# Main PyQt5 QWidget class
class Visualizer(QWidget):
    # List of NumPy images
    image_list = None

    # List of image file names, steering angles, and errors
    image_data = []

    # The image we are currently on
    image_index = 0

    # The label that displays the current image
    image_box = None

    # List of lists of points to be drawn on the line graph
    # One line for every color in the list of colors
    line_point_lists = [[] for _ in range(len(LINE_DATA))]

    # Vertical center of the line graph
    line_graph_center = None

    # Multiplier to convert a steering angle into pixels from the vertical center of the line graph
    line_graph_multiplier = None

    # The right bound of the line graph
    line_graph_right_bound = None

    # Fonts for the line graph and heat map labels
    heat_map_labels_font = None
    line_graph_labels_font = None
    line_graph_legend_font = None

    # Call various initialization functions
    def __init__(self):

        # Call the superclass initializer
        super(Visualizer, self).__init__()

        # Check that the number of command line arguments is correct
        num_arguments = len(sys.argv)
        if num_arguments != 3 and num_arguments != 4:
            print('Usage:', sys.argv[0],
                  '<image folder> <right line trained model> <left line trained model (optional)>')
            sys.exit()

        # Generate the fonts for the line graph and heat map
        self.heat_map_labels_font, self.line_graph_labels_font, self.line_graph_legend_font = [
            QFont(UI_FONT_NAME, font_size)
            for font_size in [
                HEAT_MAP_LABELS_FONT_SIZE, LINE_GRAPH_LABELS_FONT_SIZE,
                LINE_GRAPH_LEGEND_FONT_SIZE
            ]
        ]

        # Load the paths to the models and image provided as command line arguments
        image_folder = os.path.expanduser(sys.argv[1])

        # Create an inference wrapper based on the number of models provided
        num_models = num_arguments - 2
        if num_models == 1:
            inference_and_steering_wrapper = InferenceWrapperSingleLine(sys.argv[2])
        else:
            inference_and_steering_wrapper = InferenceWrapperTwoLines([sys.argv[3], sys.argv[2]])

        # Load and perform inference on the images
        self.image_list, self.image_data = process_images(
            image_folder=image_folder,
            inference_and_steering_wrapper=inference_and_steering_wrapper,
            marker_radius=MARKER_RADIUS,
            heat_map_opacity=HEAT_MAP_OPACITY,
            light_gray_color=LIGHT_GRAY_COLOR
        )

        # Set the global image height and width variables
        image_height, image_width = self.image_list[0].shape[:2]

        # Calculate the height of one vertical half of the line graph ignoring the border
        half_graph_height_minus_border = (LINE_GRAPH_HEIGHT / 2) - LINE_GRAPH_BORDER_HEIGHT

        # Use that, divided by the predefined guide line steering angle, to calculate the line graph multiplier
        self.line_graph_multiplier = int(half_graph_height_minus_border /
                                         LINE_GRAPH_GUIDE_LINE_ABSOLUTE_VALUE)

        # Set up the UI
        self.init_ui(image_height, image_width)

    # Initialize the user interface
    def init_ui(self, image_height, image_width):

        # The size of the image box is that of the original image times SCALING_FACTOR
        image_box_width = image_width * SCALING_FACTOR
        image_box_height = image_height * SCALING_FACTOR

        # Calculate the center of the line graph using the height of the image
        # plus the height of the line graph label as an upper limit of the graph
        self.line_graph_center = image_box_height + HEAT_MAP_LABELS_HEIGHT + (LINE_GRAPH_HEIGHT // 2)

        # To calculate the window size, use the image box size plus the predefined height that will
        # be occupied by the line graph, corresponding legend, and the label below the heat maps
        window_width = image_box_width
        window_height = image_box_height + HEAT_MAP_LABELS_HEIGHT + LINE_GRAPH_HEIGHT + LINE_GRAPH_LEGEND_HEIGHT

        # Set the size, position, title, and color scheme of the window
        self.setFixedSize(window_width, window_height)
        self.move(0, 0)
        self.setWindowTitle('Autonomous Driving System Visualizer')

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
        self.image_box.setFixedSize(image_box_width, image_box_height)
        self.image_box.move(0, 0)

        # Create labels below the image box that identify the two heat maps
        self.create_heat_map_labels(image_box_width, image_box_height)

        # Create numerical labels next to the line graph
        self.create_line_graph_labels()

        # Create a legend below the line graph
        self.create_line_graph_legend(image_box_width, image_box_height)

        # Make the window exist
        self.show()

        # Display the initial image
        self.update_display(1)

    # Create labels below the heat maps in the image box that identify their function
    def create_heat_map_labels(self, image_box_width, image_box_height):

        # Create two labels in a loop
        for i in range(2):
            # The width of the label will be half of the width of the main image box, rounded to an integer
            label_width = int(round(image_box_width / 2))

            # Get the X position by multiplying the width by the index
            x_position = label_width * i

            # The Y position is equal to the bottom of the main image box,
            # which is equal to its height since its Y position is zero
            y_position = image_box_height

            # Create and format the label
            heat_map_label = QLabel(self)
            heat_map_label.setFont(self.heat_map_labels_font)
            heat_map_label.setAlignment(Qt.AlignCenter)
            heat_map_label.move(x_position, y_position)
            heat_map_label.setFixedSize(label_width, HEAT_MAP_LABELS_HEIGHT)
            heat_map_label.setText(HEAT_MAP_LABELS_TEXT[i])

    # Create labels on the line graph for the steering angles at which guide lines are drawn
    def create_line_graph_labels(self):

        # Iterate over the three relevant steering angles
        for steering_angle in [-LINE_GRAPH_GUIDE_LINE_ABSOLUTE_VALUE, 0, LINE_GRAPH_GUIDE_LINE_ABSOLUTE_VALUE]:
            # Calculate the Y position at which to center the label based on the steering angle
            y_position_center = self.get_line_graph_y_position(steering_angle)

            # Offset it by half of the label size, because the coordinates correspond to the top left corner
            y_position_offset = y_position_center - \
                                (LINE_GRAPH_LABEL_SIZE // 2)

            # Create and format the label
            line_graph_label = QLabel(self)
            line_graph_label.setFont(self.line_graph_labels_font)
            line_graph_label.move(self.line_graph_right_bound,
                                  y_position_offset)
            line_graph_label.setAlignment(Qt.AlignCenter)
            line_graph_label.setFixedSize(LINE_GRAPH_LABEL_SIZE,
                                          LINE_GRAPH_LABEL_SIZE)
            line_graph_label.setText(str(steering_angle))

    # Create a legend below the line graph describing the various lines
    def create_line_graph_legend(self, image_box_width, image_box_height):

        # Create and configure a label extending to the left, right, and bottom edges of the screen,
        # and with its top edge aligned with the bottom of the bar graph, including the border area
        line_graph_legend = QLabel(self)
        line_graph_legend.setAlignment(Qt.AlignCenter)
        line_graph_legend.setFont(self.line_graph_legend_font)
        line_graph_legend.setFixedSize(image_box_width,
                                       LINE_GRAPH_LEGEND_HEIGHT)
        legend_top_edge = image_box_height + HEAT_MAP_LABELS_HEIGHT + LINE_GRAPH_HEIGHT
        line_graph_legend.move(0, legend_top_edge)

        # Create a description string that describes all of the lines
        line_descriptions = ''

        # Iterate over the configuration data for each of the lines
        for line_name, _, color_name, unit_description in LINE_DATA:
            # Format a string that describes the line, using inline color notation
            line_description = '<font color=\'' + color_name + \
                               '\'>■</font>: ' + line_name + \
                               ' (' + unit_description + ') '

            # Add it to the main description string
            line_descriptions += line_description

        # Set the text in the legend to the description string
        line_graph_legend.setText(line_descriptions)

    # Update the image in the image box
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
        current_image_qpixmap = QPixmap(current_image_qimage).scaled(
            self.image_box.width(), self.image_box.height())

        # Fill the image box with the picture
        self.image_box.setPixmap(current_image_qpixmap)

        # Iterate over the lists of points, each corresponding to one of the elements in the current
        # list of image data, not including the file name, which is the first element of the list
        for point_list, data_value, line_data in zip(self.line_point_lists, self.image_data[self.image_index][1:],
                                                     LINE_DATA):

            # Scale the data value by its corresponding multiplier
            multiplier = line_data[1]
            data_value_scaled = data_value * multiplier

            # Add the new point to the line graph five pixels left of the right edge
            y_point = self.get_line_graph_y_position(data_value_scaled)

            # Add a point to the list at the current Y position and five pixels left of the right bound of the graph
            point_list.append([self.line_graph_right_bound, y_point])

            # Shift all the points on the graph left by 5 pixels
            for point in point_list:
                point[0] -= 5

        # Force PyQt to repaint all of the lines
        self.repaint()

        # Print a blank line to separate this image from the last
        print()

        # Print the file name of the image, which is the first element of the data collection
        print('File name:', self.image_data[self.image_index][0])

        # Print some metadata about the image, with the labels provided in the line data
        for line_data, value in zip(LINE_DATA,
                                    self.image_data[self.image_index][1:]):
            # Extract the name from the line data
            name = line_data[0]

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
            -LINE_GRAPH_GUIDE_LINE_ABSOLUTE_VALUE)
        y_zero = self.get_line_graph_y_position(0)
        y_positive = self.get_line_graph_y_position(
            LINE_GRAPH_GUIDE_LINE_ABSOLUTE_VALUE)

        # Draw the three grid lines
        start_x_position = 0
        end_x_position = self.line_graph_right_bound
        paint_line(
            [[start_x_position, y_negative], [end_x_position, y_negative]],
            QColor(0, 0, 0))
        paint_line([[start_x_position, y_zero], [end_x_position, y_zero]],
                   QColor(0, 0, 0))
        paint_line(
            [[start_x_position, y_positive], [end_x_position, y_positive]],
            QColor(0, 0, 0))

        # Draw each of the lists of points on the graph with their corresponding colors
        for point_list, line_data in zip(self.line_point_lists, LINE_DATA):
            # Get the color corresponding to the name in the line data
            color_name = line_data[2]
            color = QColor()
            color.setNamedColor(color_name)
            paint_line(point_list, color)

    # Take an arbitrary steering angle, return the Y position
    # that the angle would correspond to on the graph
    def get_line_graph_y_position(self, steering_angle):
        y_point = -int(steering_angle *
                       self.line_graph_multiplier) + self.line_graph_center
        return y_point


# If this file is being run directly, instantiate the ManualSelection class
if __name__ == '__main__':
    app = QApplication([])
    ic = Visualizer()
    sys.exit(app.exec_())
