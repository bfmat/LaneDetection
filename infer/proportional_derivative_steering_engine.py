from __future__ import division

import numpy


# A system for converting line positions on each side of the road to steering angles, using outlier detection and PID.
# Created by brendon-ai, September 2017


# Main class, instantiated with PID parameters and road edge weights
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

    # Compute a steering angle. given points down the center of the road
    def compute_steering_angle(self, center_points):

        # If there are not at least two points, return None because there is no reasonable line of best fit
        if len(center_points) < 2:
            return None

        # Compute the line of best fit for the center line
        line = line_of_best_fit(center_points)

        # Calculate two points on the line at the predefined high and low positions
        center_x_high, center_x_low = [(y_position * line[1]) + line[0]
                                       for y_position in (self.center_y_high, self.center_y_low)]

        # Calculate the proportional error from the ideal center
        proportional_error = self.ideal_center_x - center_x_high

        # Calculate the derivative error which is the inverse of the slope of the line
        # Using the inverse avoids the error approaching infinity as the line becomes vertical
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


# Calculate a line of best fit for a set of points (Y, X format is assumed)
def line_of_best_fit(points):

    # Get an array of X positions from the points
    x = numpy.array([point[1] for point in points])

    # Get a list of Y positions, with a bias term of 1 at the beginning of each row
    y = numpy.array([[1, point[0]] for point in points])

    # Use the normal equation to find the line of best fit
    y_transpose = y.transpose()
    line_parameters = numpy.linalg.pinv(y_transpose.dot(y)).dot(y_transpose).dot(x)

    return line_parameters
