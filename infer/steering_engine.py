from __future__ import division

import math
import numpy


# A system for converting line positions on each side of the road to steering angles, using outlier detection and PID.
# Created by brendon-ai, September 2017


# Main class, instantiated with PID parameters, and road edge weights
class SteeringEngine:

    # Distance off of the line of best fit a point must be to be considered an outlier
    max_average_variation = None

    # Positive multiplier for the proportional error term calculated for steering
    steering_multiplier = None

    # Ideal horizontal position for the center of the road
    ideal_center_x = None

    # Vertical position at which the center of the road is calculated
    center_y = None

    # Maximum permitted absolute value for the steering angle
    steering_limit = None

    # Set global variables provided as arguments
    def __init__(self, max_average_variation, steering_multiplier, ideal_center_x, center_y, steering_limit):
        self.max_average_variation = max_average_variation
        self.steering_multiplier = steering_multiplier
        self.ideal_center_x = ideal_center_x
        self.center_y = center_y
        self.steering_limit = steering_limit

    # Compute a steering angle. given points on each road line
    def compute_steering_angle(self, all_points):

        # Remove the outliers from each set of points
        all_points_no_outliers = [remove_outliers(points, self.max_average_variation) for points in all_points]

        # If all the points were considered outliers, and the lists are empty
        for points in all_points_no_outliers:
            if not points:

                # Exit early and return None
                return None

        # Calculate the lines of best fit for each set of points
        lines_of_best_fit = [line_of_best_fit(points) for points in all_points_no_outliers]

        # Calculate the horizontal position of each line at the defined vertical position
        horizontal_positions = [(self.center_y - line[0]) / line[1] for line in lines_of_best_fit]

        # Calculate the average of the two positions, which is the center of the road
        center_x = sum(horizontal_positions) / len(horizontal_positions)

        # Calculate the error from the ideal center
        error = self.ideal_center_x - center_x

        # Multiply the error by the steering multiplier
        steering_angle = error * self.steering_multiplier

        # If the steering angle is greater than the maximum, set it to the maximum
        if steering_angle > self.steering_limit:
            steering_angle = self.steering_limit

        # If it is less than the minimum, set it to the minimum
        elif steering_angle < -self.steering_limit:
            steering_angle = -self.steering_limit

        # Return the steering angle and the error
        return steering_angle, error


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


# Calculate a line of best fit for a set of points
def line_of_best_fit(points):

    # Get an array of Y positions from the points
    y = numpy.array([point[1] for point in points])

    # Get a list of X positions, with a bias term of 1 at the beginning of each row
    x = numpy.array([[1, point[0]] for point in points])

    # Use the normal equation to find the line of best fit
    x_transpose = x.transpose()
    line_parameters = numpy.linalg.pinv(x_transpose.dot(x)).dot(x_transpose).dot(y)

    return line_parameters
