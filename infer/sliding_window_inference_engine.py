from skimage.util import view_as_windows


# A system for running inference on sections of an image using sliding windows
# Created by brendon-ai, September 2017


# A number used as a stride for the channel axis, ensuring that no window slicing occurs on that axis
MORE_THAN_CHANNELS = 6


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

        # Set scalar variables
        self.model = model
        self.window_size = window_size

        # If stride is a tuple, set the global variable to that tuple plus a term corresponding to the channel axis
        if isinstance(stride, tuple):
            self.stride = stride + (MORE_THAN_CHANNELS,)

        # If stride is a scalar, set the global variable to a three-element tuple with that value plus a channel term
        else:
            self.stride = (stride, stride, MORE_THAN_CHANNELS)

    # Given an image, compute a vector of positions describing the position of the line within each row
    def infer(self, image):

        # The distance of the very first window from the top and side of the whole image
        offset_from_side = self.window_size // 2

        # Slice up the image into windows
        image_slices = view_as_windows(image, (self.window_size, self.window_size, 3), self.stride)

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

            # If there are no values in the prediction list which are positive (greater than 0.5)
            if not sum(prediction > 0.5 for prediction in row_predictions):

                # Simply use the max value index
                prediction_index = max_prediction_index

            else:

                # List of relative horizontal positions of valid adjacent windows
                adjacent_options = []

                # If the maximum is not right against the left edge, we can search the window to the left
                if max_prediction_index > 0:
                    adjacent_options.append(-1)

                # If it is not on the right edge, search the window to the right
                if max_prediction_index < len(row_predictions) - 1:
                    adjacent_options.append(1)

                # Get the index of the greater of the two values adjacent to the maximum
                adjacent_max = max(row_predictions[max_prediction_index + i] for i in adjacent_options)
                adjacent_max_index = row_predictions.index(adjacent_max)

                # Calculate the proportions that the global max and adjacent max each contribute to the sum of the two
                total_prediction = max_prediction + adjacent_max
                adjacent_weight = adjacent_max / total_prediction
                max_weight = max_prediction / total_prediction

                # Calculate a weighted average of their indices
                prediction_index = (adjacent_max_index * adjacent_weight) + (max_prediction_index * max_weight)

            # Using the stride and window size, compute the horizontal and vertical positions corresponding to the index
            max_vertical_position = offset_from_side + (self.stride[0] * row)
            max_horizontal_position = offset_from_side + (self.stride[1] * prediction_index)

            # Make a tuple containing the overall position and add it to the list of positions
            position = (max_horizontal_position, max_vertical_position)
            line_positions.append(position)

        return line_positions
