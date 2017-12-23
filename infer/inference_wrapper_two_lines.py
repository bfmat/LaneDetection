import os

from keras.models import load_model

from .proportional_derivative_steering_engine import PDSteeringEngine
from .sliding_window_inference_engine import SlidingWindowInferenceEngine
from ..infer.lane_center_calculation import calculate_lane_center_positions_two_lines


# A wrapper that generates and uses two inference engines, one for each road line, and a steering engine,
# for running inference on an image and calculating a steering angle

# Main class, instantiated with paths to sliding window models
class InferenceWrapperTwoLines:
    # Array of two inference engines, one for each road line
    inference_engines = []

    # Steering engine for calculating steering angle and other output values
    steering_engine = None

    # Load models and create inference and steering engines
    def __init__(self, model_paths):
        # Get the models from the folder
        for model_path in model_paths:
            # Convert the home folder path to an absolute path
            absolute_path = os.path.expanduser(model_path)

            # Load the model from the absolute path
            model = load_model(absolute_path)

            # Create a sliding window inference engine with the model
            inference_engine = SlidingWindowInferenceEngine(
                model=model, slice_size=16, stride=4)

            # Add it to the list of steering engines
            self.inference_engines.append(inference_engine)

        # Instantiate the steering angle generation engine
        self.steering_engine = PDSteeringEngine(
            proportional_multiplier=0.0025,
            derivative_multiplier=0,
            max_distance_from_line=5,
            ideal_center_x=190,
            center_y=60,
            steering_limit=0.2
        )

    # Run inference using the inference and steering engines and return the output values
    def infer(self, image):
        # With each of the provided engines, perform inference
        # on the current image, calculating a prediction tensor
        prediction_tensors = [inference_engine.infer(image) for inference_engine in self.inference_engines]

        # Calculate the center line positions and add them to the list
        center_line_positions, outer_road_lines = calculate_lane_center_positions_two_lines(
            left_line_prediction_tensor=prediction_tensors[0],
            right_line_prediction_tensor=prediction_tensors[1],
            minimum_prediction_confidence=0.9,
            original_image_shape=image.shape,
            window_size=self.inference_engines[0].window_size)

        # Calculate a steering angle and errors from the points
        output_values = self.steering_engine.compute_steering_angle(center_line_positions)

        # Return the output values, along with the center and side lines as well as the line of best fit
        return output_values, center_line_positions, outer_road_lines, self.steering_engine.center_line_of_best_fit
