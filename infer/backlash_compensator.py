import numpy as np

# A persistent class that compensates for backlash based on current and past steering angles
# Created by brendon-ai, January 2018


# The factor by which to multiply the delta before adding it to the steering angle
DELTA_MULTIPLIER = 1.5

# The minimum absolute value of a delta that is large enough to be considered a change in direction
NOISE_THRESHOLD = 0.001


# Main class with a persistent state that is used to calculate future steering angles
class BacklashCompensator:

    # Initialize global variables
    def __init__(self):
        # Storage for the previous input steering angle
        self.previous_input_steering_angle = 0
        # The delta from the previous input steering angle calculated during the previous steering angle calculation
        self.previous_delta = 0

    # Process a steering angle to compensate for backlash, returning a modified steering angle
    def process(self, steering_angle):
        # Calculate the delta from the previous input steering angle
        delta = steering_angle - self.previous_input_steering_angle
        # If the sign of the current delta is different from the previous delta, that is the current steering angle
        # constitutes a change in movement direction
        # And also if the absolute value of the delta is also above a certain threshold, so that overly small deltas,
        # which can probably be considered noise, are not considered a change in direction
        if np.sign(delta) != np.sign(self.previous_delta) and abs(delta) >= NOISE_THRESHOLD:
            # Add the delta multiplied by a scaling factor to the steering angle,
            # to accelerate the change in direction and take up more of the dead band
            processed_steering_angle = steering_angle + (delta * DELTA_MULTIPLIER)
        # Otherwise, the steering angle is continuing to move in the same direction
        else:
            # Use the steering angle unmodified
            processed_steering_angle = steering_angle
        # Store the current delta and steering angle for the next time this function is called
        self.previous_input_steering_angle = steering_angle
        self.previous_delta = delta
        # Return the processed steering angle
        return processed_steering_angle
