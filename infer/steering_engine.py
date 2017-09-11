import numpy as np

# A system for converting line positions on each side of the road to steering angles, using outlier detection and PID
# Created by brendon-ai, September 2017


# Main class, instantiated with PID parameters, and road edge weights
class SteeringEngine:

    # Distance off of the line of best fit a point must be to be considered an outlier
    max_line_variation = None

    # Y position at which the road center point is calculated
    center_point_height = None

    # Set global variables provided as arguments
    def __init__(self, max_line_variation, center_point_height):
        self.max_line_variation = max_line_variation
        self.center_point_height = center_point_height

    # Compute a steering angle. given points on each road line
    def compute_steering_angle(self, left_points, right_points):

        # Calculate the lines of best fit
        left_line, right_line = (line_of_best_fit(points) for points in (left_points, right_points))

        # Remove the outliers from each line
        left_points_no_outliers = remove_outliers(left_points, left_line, self.max_line_variation)
        right_points_no_outliers = remove_outliers(right_points, right_line, self.max_line_variation)

        # Recalculate the lines of best fit
        lines_no_outliers = (line_of_best_fit(points) for points in (left_points_no_outliers, right_points_no_outliers))

        # Calculate the average of the two lines, that is, the center line of the road
        center_line = [(a + b) / 2 for a, b in lines_no_outliers]

        # Find the horizontal position of the center line at the given vertical position
        # center_x = (self.center_point_height - center_line[1]) / center_line[0]

        # PID stuff
        # TODO


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
def remove_outliers(points, line, max_variation):

    # List with outliers removed that we will return
    output_points = []

    # Calculate the square of the maximum permitted variation
    sqr_max_variation = max_variation ** 2

    # Calculate the perpendicular slope to the line
    perpendicular_slope = 1 / line[1]

    # Iterate over each of the points
    for point in points:

        # Find the distance from this particular point to its projection on the line
        # Calculate where the point's X value is on the perpendicular line with Y intercept 0
        predicted_y = point[0] * perpendicular_slope

        # Use that to compute the bias term and calculate the perpendicular line that passes through the point
        y_intercept = point[1] - predicted_y

        # Calculate the intersection point of the lines
        # Start by subtracting one line from the other
        line_minus_perpendicular_line = (line[0] - y_intercept, line[1] - perpendicular_slope)

        # Find the value of X so that the corresponding Y value on this new line is 0
        intersection_x = -line_minus_perpendicular_line[0] / line_minus_perpendicular_line[1]

        # Now calculate the Y value of this intersection point and wrap the two values in a tuple
        intersection = (intersection_x, line[0] + (line[1] * intersection_x))

        # This intersection point is the projection of the original point onto the line
        # Calculate its Pythagorean distance from the original point
        sqr_distance = sum(((a - b) ** 2 for a, b in zip(point, intersection)))

        # If the distance is less than or equal to the maximum permitted variation, add it to the output list
        if sqr_distance <= sqr_max_variation:
            output_points.append(point)

    return output_points
