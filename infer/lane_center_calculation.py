from __future__ import division

import math


# A system for calculating points on a line down the center of the lane based on a tensor of predictions for each window
# Created by brendon-ai, October 2017


# Calculate the center line of a tensor of predictions of an arbitrary size, with a minimum confidence for the line
# and scale it so that the output maps onto locations in the original source image
def calculate_lane_center_positions(prediction_tensor, minimum_prediction_confidence,
                                    original_image_width, original_window_width):

    # Add the center points of the rows to a list
    center_positions = []

    # Iterate over the rows
    for row in prediction_tensor:

        # Find the peak in both directions from the last center position
        peak_indices = [find_peak_in_direction
                        (row, center_positions[-1], reversed_iteration_direction, minimum_prediction_confidence)
                        for reversed_iteration_direction in (False, True)]

        # If a peak could be found in both directions
        if None not in peak_indices:

            # Calculate the average of the two peaks
            center_position = sum(peak_indices) / len(peak_indices)

            # Scale and offset it so that it corresponds to the correct position within the original image
            width_scaling_factor =
            center_position_scaled = (center_position * width_scaling_factor) - original_window_width

            center_positions.append(center_position)

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


# Remove all outliers from a list of points given a line of best fit
def remove_outliers(points, max_variation):

    # List with outliers removed that we will return
    output_points = []

    # Iterate over each of the points
    for point in points:

        # Number of points that this point was within the permitted distance of
        valid_comparisons = 0

        # Iterate over each of the points again
        for comparison_point in points:

            # Calculate the Pythagorean distance between the two points
            distance = math.sqrt(((point[0] - comparison_point[0]) ** 2) + ((point[1] - comparison_point[1]) ** 2))

            # If the value is within the permitted maximum
            if distance < max_variation:

                # Increment the counter for this point
                valid_comparisons += 1

        # If this point was within the required distance of at least three points including itself (so two others)
        if valid_comparisons >= 3:
            output_points.append(point)

    return output_points
