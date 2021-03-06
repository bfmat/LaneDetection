from __future__ import division

# A system for calculating points on a line down the center of the lane based on a tensor of predictions for each
# sliding window system, using an algorithm that traverses up the lane calculating the center on each iteration
# Created by brendon-ai, October 2017

# Value by which the offset of the search starting position is increased each time road edges could not be found
STARTING_POSITION_OFFSET_INCREMENT = 5

# Maximum absolute value that the offset can have before we give up and break out of the loop
STARTING_POSITION_OFFSET_MAXIMUM = 50


# Calculate the center line of a two tensors of predictions of arbitrary size, with a minimum confidence for the line,
# and scale it so that the output maps onto locations in the original source image
def calculate_lane_center_positions_two_lines(
        left_line_prediction_tensor, right_line_prediction_tensor,
        minimum_prediction_confidence, original_image_shape, window_size):
    # Add the center points of the rows to a list
    center_positions = []

    # Record the center position every iteration to use as the starting position later
    # Use the center of the image as an initial value
    starting_position = left_line_prediction_tensor.shape[1] // 2

    # List of lists of corresponding positions on two road lines, which must be transposed before external use
    all_corresponding_outer_positions = []

    # Iterate over the rows backwards, going from the bottom to the top
    for y_position in range(len(left_line_prediction_tensor) - 1, -1, -1):

        # Initialize the road edge indices to a list containing None
        peak_indices = [None]

        # We will gradually increase the magnitude of the offset of the starting position until road edges are found
        starting_position_offset = 0

        # Loop until two road edges have been found
        while None in peak_indices:

            # Try to find the peak in both directions from the last center position
            peak_indices = [
                find_peak_in_direction(prediction_tensor[y_position],
                                       starting_position,
                                       reversed_iteration_direction,
                                       minimum_prediction_confidence)
                for prediction_tensor, reversed_iteration_direction in zip((
                    left_line_prediction_tensor,
                    right_line_prediction_tensor), (True, False))
            ]

            # If the offset is currently positive or zero, increment it
            if starting_position_offset >= 0:
                starting_position_offset += STARTING_POSITION_OFFSET_INCREMENT

            # Invert the sign of the offset on every iteration so we will try every offset within an increasing range
            starting_position_offset *= -1

            # Break out of the loop if the offset has become unreasonably large
            # If this happens, the image probably has no valid starting position so we will just give up
            if abs(starting_position_offset) > STARTING_POSITION_OFFSET_MAXIMUM:
                break

        # If a valid solution has been found (we haven't broken out due to a large offset)
        if None not in peak_indices:
            # Add the points on the two peaks to a list after combining them with Y positions
            corresponding_outer_positions = [
                scale_position(
                    position=(y_position, peak_index),
                    original_image_shape=original_image_shape,
                    prediction_tensor_shape=left_line_prediction_tensor.shape,
                    window_size=window_size
                )
                for peak_index in peak_indices
            ]
            all_corresponding_outer_positions.append(
                corresponding_outer_positions)

            # Calculate the average of the two peaks and add the Y position of the row to the tuple
            center_x_position = sum(peak_indices) / len(peak_indices)
            center_position = (y_position, center_x_position)

            # Use the center position, rounded to an integer, as the starting position
            starting_position = int(center_position[1])

            # Scale the position and add it to the list
            center_position_processed = scale_position(
                position=center_position,
                original_image_shape=original_image_shape,
                prediction_tensor_shape=left_line_prediction_tensor.shape,
                window_size=window_size
            )
            center_positions.append(center_position_processed)

    # Transpose the list of points in the outer lines
    outer_road_lines = zip(*all_corresponding_outer_positions)

    # Return both the center line and the outer lines
    return center_positions, outer_road_lines


# Find a single road line and compute the center line of the road by offsetting the points on the right line
def calculate_lane_center_positions_single_line(prediction_tensor, original_image_shape, window_size,
                                                minimum_prediction_confidence, offset_multiplier, offset_absolute, search_only_in_bottom_portion):
    # A list to hold the points on the approximated center line
    center_line_points = []
    # A list to hold the points on the outer line
    outer_road_line = []
    # If required, search only in 75% of the rows of the prediction tensor (so that the sky is not included)
    num_rows = len(prediction_tensor)
    if search_only_in_bottom_portion:
        starting_row = int(num_rows * 0.25)
        search_range = range(starting_row, num_rows)
    # Otherwise, search the entire image
    else:
        search_range = range(num_rows)
    # Iterate over the range, dropping points
    for row_index in search_range:
        # Get the value of the row itself
        row = prediction_tensor[row_index]
        # Try to find the first peak in this row to the right of the center
        right_peak = find_peak_in_direction(
            collection=row,
            starting_index=len(row) // 2,
            reversed_iteration_direction=False,
            minimum_value=minimum_prediction_confidence
        )
        # If a peak has been found
        if right_peak is not None:
            # Scale the peak to a point on the image
            scaled_peak = scale_position((row_index, right_peak), original_image_shape, prediction_tensor.shape,
                                         window_size)
            # Add the peak to the list of outer line points
            outer_road_line.append(scaled_peak)
            # Subtract Y position times a constant from the X position to offset it because the center line gradually
            # becomes closer to the the right line further up the image
            # Also subtract an unmodified constant to shift the line further into the center of the road
            center_x_position = scaled_peak[1] - \
                (scaled_peak[0] * offset_multiplier) - offset_absolute
            # Add the position to the list
            center_line_points.append((scaled_peak[0], center_x_position))
    # Return the list of points and the outer line
    return center_line_points, [outer_road_line]


# A function to traverse a collection from an arbitrary point to the end, finding the first value above a certain
# threshold and continuing until the first value which drops below that threshold is found, finding a local maximum
# and returning the synthetic interpolated list index of that peak
def find_peak_in_direction(collection, starting_index,
                           reversed_iteration_direction, minimum_value):
    # Storage for the indices of the first value that passed the threshold
    initial_sufficient_value_index = None

    # Iterate over the row, starting at the center and continuing to the end in the provided direction
    ending_index = 0 if reversed_iteration_direction else len(collection) - 1
    iteration_step = -1 if reversed_iteration_direction else 1
    for i in range(starting_index, ending_index, iteration_step):

        # If the current value is greater than or equal to the threshold, record its index and stop iterating
        if collection[i] >= minimum_value:
            initial_sufficient_value_index = i
            break

    # Return the index, which will be None if a sufficient value could not be found
    return initial_sufficient_value_index


# A function to process a single point and scale it so that it corresponds to a position on the original image
def scale_position(position, original_image_shape, prediction_tensor_shape, window_size):
    # Scale and offset the point so that it corresponds to the correct position within the original image
    center_position_scaled = [
        center_position_element *
        (image_shape_element / prediction_tensor_shape_element)
        for center_position_element, image_shape_element, prediction_tensor_shape_element
        in zip(position, original_image_shape, prediction_tensor_shape)
    ]
    center_position_offset = [
        element + (window_size // 2) for element in center_position_scaled
    ]

    # Round the processed position to an integer and return it
    return [int(value) for value in center_position_offset]
