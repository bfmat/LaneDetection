import math

# A system for converting line positions on each side of the road to steering angles, using outlier detection and PID
# Created by brendon-ai, September 2017


# Main class, instantiated with PID parameters, and road edge weights
class SteeringEngine:

    # Distance off of the line of best fit a point must be to be considered an outlier
    max_average_variation = None

    # Ideal horizontal position for the center of the road
    ideal_center_x = None

    # Positive multiplier for the proportional error term calculated for steering
    steering_multiplier = None

    # Maximum permitted absolute value for the steering angle
    steering_limit = None

    # Set global variables provided as arguments
    def __init__(self, max_average_variation, steering_multiplier, ideal_center_x, steering_limit):
        self.max_average_variation = max_average_variation
        self.steering_multiplier = steering_multiplier
        self.ideal_center_x = ideal_center_x
        self.steering_limit = steering_limit

    # Compute a steering angle. given points on each road line
    def compute_steering_angle(self, left_points, right_points):

        # Remove the outliers from each set of points
        left_points_no_outliers = remove_outliers(left_points, self.max_average_variation)
        right_points_no_outliers = remove_outliers(right_points, self.max_average_variation)

        # If all the points were considered outliers, and the lists are empty
        if not left_points_no_outliers or not right_points_no_outliers:

            # Exit early and return None
            return None

        # Calculate the average horizontal positions of the remaining points in each line
        left_average = sum([position[0] for position in left_points_no_outliers]) / len(left_points_no_outliers)
        right_average = sum([position[0] for position in right_points_no_outliers]) / len(right_points_no_outliers)

        # Calculate the average of the two points, that is, the center of the road
        center_x = (left_average + right_average) // 2

        # Calculate the error from the ideal center
        error = self.ideal_center_x - center_x

        # Multiply the error by the steering multiplier
        steering_angle = error * self.steering_multiplier

        # If the steering angle is greater than the maximum, set it to the maximum
        if steering_angle > self.steering_limit:
            steering_angle = steering_limit

        # If it is less than the minimum, set it to the minimum
        elif steering_angle < -self.steering_limit:
            steering_angle = -steering_limit

        return steering_angle


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
