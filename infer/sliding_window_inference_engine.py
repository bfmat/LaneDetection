from time import time
from skimage.util import view_as_windows


# A system for running inference on sections of an image using sliding windows
# Created by brendon-ai, September 2017


# Main class, instantiated with configuration and trained model
class SlidingWindowInferenceEngine:

    # The Keras model used for inference on square sections of the image
    model = None

    # The side length of the sliding windows
    window_size = None

    # The size of gaps between positions where the sliding windows are used
    stride = None

    # Set global variables provided as arguments
    def __init__(self, model, window_size, stride):
        self.model = model
        self.window_size = window_size
        self.stride = stride

    # Given an image, compute a vector of positions describing the position of the line within each row
    def infer(self, image):

        # Starting time
        print('Started at ' + str(time()))

        # The distance of the very first window from the top and side of the whole image
        offset_from_side = self.window_size // 2

        # Slice up the image into windows
        image_slices = view_as_windows(image, (self.window_size, self.window_size, 3), self.stride)

        # Viewing as windows complete
        print('Viewing as windows complete at ' + str(time()))

        # A two-dimensional list that will contain the line positions for each row
        line_positions = []

        # Loop over the windows and classify them, one row at a time
        for row in range(image_slices.shape[0]):

            # Create a one-dimensional list of the predictions for this row
            row_predictions = []

            # Loop over the second dimension
            for column in range(image_slices.shape[1]):

                # Slice the individual window out of the list
                window = image_slices[row, column]

                # Use the model to classify the image, and convert the result to a Boolean
                prediction = self.model.predict(window)[0, 0]

                # Append it to the list of classifications for this row
                row_predictions.append(prediction)

            # We must now compute the position of the line based on the list of classifications for the row
            # Find the max value in the list (the most likely to be a line)
            max_prediction = max(row_predictions)

            # Find the index of the max value
            max_prediction_index = row_predictions.index(max_prediction)

            # Using the stride and window size, compute the horizontal and vertical positions corresponding to the index
            max_vertical_position = offset_from_side + (self.stride * row)
            max_horizontal_position = offset_from_side + (self.stride * max_prediction_index)

            # Make a tuple containing the overall position
            position = (max_horizontal_position, max_vertical_position)

            # Add it to the list of positions
            line_positions.append(position)

        # Completion time
        print('Complete at ' + str(time()))

        return line_positions
