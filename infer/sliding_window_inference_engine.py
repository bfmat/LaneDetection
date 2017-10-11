from __future__ import division

import numpy

from skimage.util import view_as_windows


# A system for efficiently running inference on sections of an image using sliding windows and calculating points down
# the center of the road based on the inference output
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

    # Set-and-forget externally accessible storage for the prediction tensor of the most recently processed image
    last_prediction_tensor = None

    # Set global variables provided as arguments
    def __init__(self, model, slice_size, stride):

        # Set scalar variables
        self.model = model
        self.window_size = slice_size

        # If stride is a tuple, set the global variable to that tuple plus a term corresponding to the channel axis
        if isinstance(stride, tuple):
            self.stride = stride + (MORE_THAN_CHANNELS,)

        # If stride is a scalar, set the global variable to a three-element tuple with that value plus a channel term
        else:
            self.stride = (stride, stride, MORE_THAN_CHANNELS)

    # Given an image, compute a vector of positions describing the position of the line within each row
    def infer(self, image):

        # Slice up the image into windows
        image_slices = view_as_windows(image, (self.window_size, self.window_size, 3), self.stride)

        # Flatten the window array so that there is only one non-window dimension
        image_slices_flat = numpy.reshape(image_slices, (-1,) + image_slices.shape[-3:])

        # Run a prediction on all of the windows at once
        predictions = self.model.predict(image_slices_flat)

        # Reshape the predictions to have the same initial two dimensions as the original list of image slices
        predictions_row_arranged = numpy.reshape(predictions, image_slices.shape[:2])

        # Save the prediction tensor so external scripts can access it if desired
        self.last_prediction_tensor = predictions_row_arranged

        # Calculate the center positions with the prediction tensor
        lane_center_positions = calculate_lane_center_positions(
            prediction_tensor=predictions_row_arranged,
            minimum_prediction_confidence=0.7,
            original_image_shape=image.shape,
            window_size=self.window_size
        )

        return lane_center_positions


# Calculate the center line of a tensor of predictions of an arbitrary size, with a minimum confidence for the line
# and scale it so that the output maps onto locations in the original source image
def calculate_lane_center_positions(prediction_tensor, minimum_prediction_confidence,
                                    original_image_shape, window_size):

    # Add the center points of the rows to a list
    center_positions = []

    # Iterate over the rows
    for y_position in range(len(prediction_tensor)):

        # Find the peak in both directions from the last center position
        peak_indices = [find_peak_in_direction
                        (prediction_tensor[y_position], center_positions[-1], reversed_iteration_direction, minimum_prediction_confidence)
                        for reversed_iteration_direction in (False, True)]

        # If a peak could be found in both directions
        if None not in peak_indices:

            # Calculate the average of the two peaks and add the Y position of the row to the tuple
            center_x_position = sum(peak_indices) / len(peak_indices)
            center_position = (y_position, center_x_position)

            # Scale and offset it so that it corresponds to the correct position within the original image
            center_position_scaled = [center_position_element * (image_shape_element / prediction_tensor_shape_element)
                                      for center_position_element, image_shape_element, prediction_tensor_shape_element
                                      in zip(center_position, original_image_shape, prediction_tensor.shape)]
            center_position_offset = [element + (window_size // 2) for element in center_position_scaled]

            # Add the processed position to the list
            center_positions.append(center_position_offset)

    return center_positions


# A function to traverse a collection from an arbitrary point to the end, finding the first value above a certain
# threshold and continuing until the first value which drops below that threshold is found, finding a local maximum
# and returning the synthetic interpolated list index of that peak
def find_peak_in_direction(collection, starting_index, reversed_iteration_direction, minimum_value):

    # Storage for the indices of the first and last values that passed the threshold
    initial_sufficient_value_index = None
    final_sufficient_value_index = None

    # Iterate over the row, starting at the center and continuing to the end in the provided direction
    ending_index = len(collection) - 1 if reversed_iteration_direction else 0
    iteration_step = -1 if reversed_iteration_direction else 1
    for i in range(starting_index, ending_index, iteration_step):

        # Get the value of the collection corresponding to the current index
        current_element = collection[i]

        # If the value is greater than or equal to the threshold and it is the first such value, record its index
        if current_element >= minimum_value and initial_sufficient_value_index is None:
            initial_sufficient_value_index = i

        # If the value is less than the threshold and there have already been values greater than it
        elif current_element < minimum_value and initial_sufficient_value_index is not None:

            # Set the final index to the current index and break out of the loop
            final_sufficient_value_index = i
            break

    # If a peak has not been found, simply return None
    if initial_sufficient_value_index is None:
        return None

    # Otherwise, return the average of the two indices, rounded to the nearest integer
    else:
        peak_center = int(round((initial_sufficient_value_index + final_sufficient_value_index) / 2))
        return peak_center
