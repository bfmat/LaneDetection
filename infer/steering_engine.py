import math
import numpy as np

# A system for converting line positions on each side of the road to steering angles, using outlier detection and PID
# Created by brendon-ai, September 2017


# Main class, instantiated with PID parameters, and road edge weights
class SteeringEngine:

    # Distance off of the line of best fit a point must be to be considered an outlier
    max_average_variation = None

    # Y position at which the road center point is calculated
    center_point_height = None

    # Ideal horizontal position for the center of the road
    ideal_center_x = None

    # Positive multiplier for the proportional error term calculated for steering
    steering_multiplier = None

    # Set global variables provided as arguments
    def __init__(self, max_average_variation, steering_multiplier, ideal_center_x, center_point_height):
        self.max_average_variation = max_average_variation
        self.steering_multiplier = steering_multiplier
        self.ideal_center_x = ideal_center_x
        self.center_point_height = center_point_height

    # Compute a steering angle. given points on each road line
    def compute_steering_angle(self, left_points, right_points):

        # Remove the outliers from each set of points
        left_points_no_outliers = remove_outliers(left_points, self.max_average_variation)
        right_points_no_outliers = remove_outliers(right_points, self.max_average_variation)

        # If all the points were considered outliers, and the lists are empty
        if not left_points_no_outliers or not right_points_no_outliers:

            # Exit early and return None
            return None

        # Recalculate the lines of best fit
        left_line, right_line = \
            (line_of_best_fit(points) for points in (left_points_no_outliers, right_points_no_outliers))

        # Calculate the transpose of the list of two lines, so the corresponding elements are matched up
        lines_no_outliers_transpose = list(map(list, zip(left_line, right_line)))

        # Calculate the average of the two lines, that is, the center line of the road
        center_line = [(a + b) / 2 for a, b in lines_no_outliers_transpose]

        # Find the horizontal position of the center line at the given vertical position
        center_x = (self.center_point_height - center_line[0]) / center_line[1]

        # Calculate the error from the ideal center
        error = self.ideal_center_x - center_x

        # Multiply the error by the steering multiplier
        steering_angle = error * self.steering_multiplier

        return left_line, right_line


# Calculate a line of best fit for a set of points
def line_of_best_fit(points):

    # Get an array of Y positions from the points
    y = np.array([point[1] for point in points])

    # Get a list of X positions, with a bias term of 1 at the beginning of each row
    x = np.array([[1, point[0]] for point in points])

    # Use the normal equation to find the line of best fit
    x_transpose = x.transpose()
    line_parameters = np.linalg.pinv(x_transpose.dot(x)).dot(x_transpose).dot(y)

    return line_parameters


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
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(point, comparison_point)))

            # If the value is within the permitted maximum
            if distance < max_variation:

                # Increment the counter for this point
                valid_comparisons += 1

        # If this point was within the required distance of at least two points
        if valid_comparisons >= 2:
            output_points.append(point)

    return output_points
