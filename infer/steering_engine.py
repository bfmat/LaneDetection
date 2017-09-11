import numpy as np

# A system for converting line positions on each side of the road to steering angles, using outlier detection and PID
# Created by brendon-ai, September 2017


# Main class, instantiated with PID parameters, and road edge weights
class SteeringEngine:

    # Distance off of the line of best fit a point must be to be considered an outlier
    max_line_variation = None

    # Set global variables provided as arguments
    def __init__(self, max_line_variation):
        self.max_line_variation = max_line_variation

    # Compute a steering angle. given points on each road line
    def compute_steering_angle(self, left_points, right_points):

        # Calculate the lines of best fit
        left_line, right_line = [line_of_best_fit(points) for points in (left_points, right_points)]

        # For eac


# Calculate a line of best fit for a set of points
def line_of_best_fit(points):

    # Get an array of Y positions from the points
    y = np.array([point[1] for point in points])

    # Get a list of X positions, with a bias term of 1 at the beginning of each row
    x = np.array([[1, point[0]] for point in points])

    # Use the normal equation to find the line of best fit
    line_parameters = np.linalg.pinv(x.transpose().dot(x)).dot(x.transpose()).dot(y)

    return line_parameters

# Remove all outliers from a list of points given a line of best fit
def remove_outliers(points, line):

    # Calculate the perpendicular slope to the line
    perpendicular_slope = 1 / line[1]

    # Iterate over each of the points
    for point in points:

        # Calculate where the point would be on the perpendicular line with Y intercept 0
        predicted_y = point[0] * perpendicular_slope

        # Use that to compute the bias term and calculate the perpendicular line that passes through the point
        y_intercept = point[1] - predicted_y

        # Calculate the intersection point of the lines
        # TODO


