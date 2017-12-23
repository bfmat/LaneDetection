import os

from keras.models import load_model

from ..infer.lane_center_calculation import calculate_lane_center_positions_single_line
from ..infer.proportional_derivative_steering_engine import PDSteeringEngine
from ..infer.sliding_window_inference_engine import SlidingWindowInferenceEngine


# A wrapper that uses a single steering engine and adds a predefined offset to points along the right line of the road
# to approximate points on the center line, which are used for steering
# Created by brendon-ai, December 2017

# Main class, instantiated with paths to sliding window models
class InferenceWrapperSingleLine:
    # Inference engine for sliding window processing of images
    inference_engines = []

    # Steering engine for calculating steering angle and other output values
    steering_engine = None

    # Load model and create inference and steering engines
    def __init__(self, model_path):
        # Convert the home folder path to an absolute path
        absolute_path = os.path.expanduser(model_path)

        # Load the model from the absolute path
        model = load_model(absolute_path)

        # Create a sliding window inference engine with the model
        self.inference_engines.append(
            SlidingWindowInferenceEngine(
                model=model,
                slice_size=16,
                stride=4
            )
        )

        # Instantiate the steering angle generation engine
        self.steering_engine = PDSteeringEngine(
            proportional_multiplier=0.0025,
            derivative_multiplier=0,
            max_distance_from_line=5,
            ideal_center_x=190,
            center_y=60,
            steering_limit=0.2
        )

    # Run inference using the predefined model and steering engine
    def infer(self, image):
        # Run inference on the image and collect a prediction tensor
        prediction_tensor = self.inference_engines[0].infer(image)
        # Calculate positions on the center line using the prediction tensor
        center_line_positions, outer_road_lines = calculate_lane_center_positions_single_line(
            prediction_tensor=prediction_tensor,
            original_image_shape=image.shape,
            window_size=self.inference_engines[0].window_size,
            minimum_prediction_confidence=0.7,
            offset_multiplier=1
        )
        # Use the steering engine to calculate a steering angle based on the center line
        output_values = self.steering_engine.compute_steering_angle(center_line_positions)

        # Return the output values, along with the center and side lines as well as the line of best fit
        return output_values, center_line_positions, outer_road_lines, self.steering_engine.center_line_of_best_fit
