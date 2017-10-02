import numpy as np

from skimage.util import view_as_windows


# A system for slicing an images and performing object detection inference on the slices
# Created by brendon-ai, September 2017


# Main class, instantiated with configuration and trained model
class WideSliceInferenceEngine:

    # Convolutional neural network model that accepts the entire wide image
    model = None

    # Dimensions of slices to take out of the main image
    slice_dimensions = None

    # Vertical distance between the wide slices the image is cut into
    vertical_stride = None

    # Set the model, slice height and vertical stride provided as an argument
    def __init__(self, model, slice_dimensions, vertical_stride):

        # Set global variables
        self.model = model
        self.slice_dimensions = slice_dimensions
        self.vertical_stride = vertical_stride

    # Given an image, compute a vector of positions describing the position of the line within each row
    def infer(self, image):

        # Split the image into wide slices
        window_size = self.slice_dimensions[0]
        window_slices = view_as_windows(image, (window_size, window_size, 3), window_size)

        # A list that will contain the line positions for each row
        line_positions = []

        # Iterate over the windows in the slice
        for i in range(len(window_slices)):

            # Convert the windows in the second dimension to a list
            windows_list = [window for window in window_slices[i]]

            # Use the network to calculate the position of the object in the image
            network_output = self.model.predict(windows_list)[0, 0]

            # Scale the network's output back into the range of the image width
            slice_width = self.slice_dimensions[1]
            horizontal_position = (network_output * slice_width) + (slice_width / 2)

            # Calculate the vertical position using the slice height
            slice_height = self.slice_dimensions[0]
            vertical_position = (i + 0.5) * slice_height

            # Compose a position tuple and add it to the list for return
            position = (horizontal_position, vertical_position)
            line_positions.append(position)

        return line_positions
