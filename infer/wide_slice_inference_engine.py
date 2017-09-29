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
        image_slices = view_as_windows(image, self.slice_dimensions, self.vertical_stride)

        # A list that will contain the line positions for each row
        line_positions = []

        # Iterate over the slices
        for i in range(len(image_slices)):

            # Remove the first useless dimension from the image
            image_slice_squeezed = np.squeeze(image_slices[i], axis=0)

            # Use the network to calculate the position of the object in the image
            horizontal_position = self.model.predict(image_slice_squeezed)[0, 0]

            # Calculate the vertical position using the slice height
            slice_height = self.slice_dimensions[0]
            vertical_position = (i + 0.5) * slice_height

            # Compose a position tuple and add it to the list for return
            position = (horizontal_position, vertical_position)
            line_positions.append(position)

        return line_positions
