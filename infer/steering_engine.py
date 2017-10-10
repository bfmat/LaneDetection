from __future__ import division

import math
import numpy


# A system for converting line positions on each side of the road to steering angles, using outlier detection and PID.
# Created by brendon-ai, September 2017


# Main class, instantiated with PID parameters, and road edge weights
class SteeringEngine:

    # Distance off of the line of best fit a point must be to be considered an outlier
    max_average_variation = None

    # Positive multipliers for the proportional and derivative error terms calculated for steering
    proportional_multiplier = None
    derivative_multiplier = None

    # Ideal horizontal position for the center of the road
    ideal_center_x = None

    # Vertical positions at which the center of the road is calculated
    # Having two allows for calculation of the derivative term
    center_y_high = None
    center_y_low = None

    # Maximum permitted absolute value for the steering angle
    steering_limit = None

    # Set global variables provided as arguments
    def __init__(self, max_average_variation, proportional_multiplier, derivative_multiplier,
                 ideal_center_x, center_y_high, center_y_low, steering_limit):
        self.max_average_variation = max_average_variation
        self.proportional_multiplier = proportional_multiplier
        self.derivative_multiplier = derivative_multiplier
        self.ideal_center_x = ideal_center_x
        self.center_y_high = center_y_high
        self.center_y_low = center_y_low
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

        # Calculate the horizontal position of each line at the defined vertical position, for both vertical positions
        all_horizontal_positions = [[(center_y - line[0]) / line[1] for line in lines_of_best_fit]
                                    for center_y in (self.center_y_high, self.center_y_low)]

        # Calculate the average of the two positions, which is the center of the road
        center_x_all = [sum(horizontal_positions) / len(horizontal_positions)
                        for horizontal_positions in all_horizontal_positions]

        # Calculate the proportional error from the ideal center
        center_x_high = center_x_all[0]
        proportional_error = self.ideal_center_x - center_x_high

        # Calculate the derivative error which is the inverse of the slope of the line
        # Using the inverse avoids the error approaching infinity as the line becomes vertical
        center_x_low = center_x_all[1]
        derivative_error = (center_x_high - center_x_low) / (self.center_y_high - self.center_y_low)

        # Multiply the error by the steering multiplier
        steering_angle = (proportional_error * self.proportional_multiplier)\
                         + (derivative_error * self.derivative_multiplier)

        # If the steering angle is greater than the maximum, set it to the maximum
        if steering_angle > self.steering_limit:
            steering_angle = self.steering_limit

        # If it is less than the minimum, set it to the minimum
        elif steering_angle < -self.steering_limit:
            steering_angle = -self.steering_limit

        # Return the steering angle and the error
        return steering_angle


