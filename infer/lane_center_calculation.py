from __future__ import division


# A system for calculating points on a line down the center of the lane based on a tensor of predictions for each
# sliding window system, using an algorithm that traverses up the lane calculating the center on each iteration
# Created by brendon-ai, October 2017


# Calculate the center line of a tensor of predictions of an arbitrary size, with a minimum confidence for the line
# and scale it so that the output maps onto locations in the original source image
def calculate_lane_center_positions(left_line_prediction_tensor, right_line_prediction_tensor,
                                    minimum_prediction_confidence, original_image_shape, window_size):

    # Add the center points of the rows to a list
    center_positions = []

    # Record the center position every iteration to use as the starting position later
    # Use the center of the image as an initial value
    starting_position = left_line_prediction_tensor.shape[1] // 2

    # Iterate over the rows backwards, going from the bottom to the top
    for y_position in range(len(left_line_prediction_tensor) - 1, -1, -1):

        # Find the peak in both directions from the last center position
        peak_indices = [find_peak_in_direction(prediction_tensor[y_position], starting_position,
                        reversed_iteration_direction, minimum_prediction_confidence)
                        for prediction_tensor, reversed_iteration_direction
                        in zip((left_line_prediction_tensor, right_line_prediction_tensor), (True, False))]

        # If a peak could be found in both directions
        if None not in peak_indices:

            # Calculate the average of the two peaks and add the Y position of the row to the tuple
            center_x_position = int(sum(peak_indices) / len(peak_indices))
            center_position = (y_position, center_x_position)

            # Use the unmodified center position as the starting position
            starting_position = center_position[1]

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
    ending_index = 0 if reversed_iteration_direction else len(collection) - 1
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
