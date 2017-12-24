from __future__ import division, print_function

import math
import os

import numpy
from numpy.linalg.linalg import LinAlgError
from scipy.misc import imread, imresize
from skimage.draw import line

from ..visualizer.errors import UnusableImageError

# A collection of functions required for loading and processing the data for the visualizer
# including many of the elements that appear to be part of the interface but really are image
# modifications, such as line markers, center lines, and the network prediction heat map
# In addition, calculate the geometric mean of the error and of the derivative
# of the error, using the bottom of the image instead of the top to avoid
# extreme fluctuation, and print out the resulting information
# Created by brendon-ai, October 2017

# List of descriptions output before their corresponding error values
ERROR_DESCRIPTIONS = [
    'Standard deviation of the position of the car with respect to the center of the road:',
    'Standard deviation of the apparent slope of the center of the road with respect to the car:'
]

# Get the expected number of error values
num_errors = len(ERROR_DESCRIPTIONS)

# List of accumulators for the squared proportional error and line slope
accumulators = [0] * num_errors

# The number of files we are loading
num_files = None


# Load and process the image with the provided inference engines and steering engine
def process_images(image_folder, inference_and_steering_wrapper, marker_radius, heat_map_opacity, light_gray_color):
    # Notify the user that we have started loading the images
    print('Loading images...')

    # List that we will add processed images to
    image_list = []

    # List of file names of images
    image_names = sorted(os.listdir(image_folder))

    # Set the global value containing the number of files
    global num_files
    num_files = len(image_names)

    # List of image data
    all_image_data = []

    # Loop over each of the images in the folder
    for image_name in image_names:

        # Load the image from disk, using its fully qualified path
        image_path = image_folder + '/' + image_name
        image = imread(image_path)

        # Try to process the image and add various markings to it, recording metadata returned for display purposes
        try:
            processed_image, output_values = _process_single_image(
                image=image,
                inference_and_steering_wrapper=inference_and_steering_wrapper,
                marker_radius=marker_radius,
                heat_map_opacity=heat_map_opacity,
                light_gray_color=light_gray_color
            )
        # If a problem is encountered, skip this image and print an error message
        except (UnusableImageError, LinAlgError):
            print('Failed to load', image_name)
            continue

        # Add the prepared image and the steering angle to their corresponding lists
        image_list.append(processed_image)

        # Add the corresponding steering angle to the data list
        all_image_data.append((image_name,) + output_values)

        # Print out the file name of the image
        print('Loaded', image_name)

    # Notify the user that loading is complete
    print('Loading complete!')

    # Calculate and print the standard deviation with respect to both errors, one at a time
    for accumulator, error_description in zip(accumulators,
                                              ERROR_DESCRIPTIONS):
        global num_files
        variance = accumulator / num_files
        standard_deviation = math.sqrt(variance)
        print(error_description, standard_deviation)

    # Return the images and their corresponding data
    return image_list, all_image_data


# Perform all necessary processing on a single image to prepare it for visualization
def _process_single_image(image, inference_and_steering_wrapper, marker_radius, heat_map_opacity, light_gray_color):
    # Perform inference on the image using the wrapper
    output_values, center_line_positions, outer_road_lines = inference_and_steering_wrapper.infer(image)[:3]

    # If None was returned, throw an error
    if output_values is None:
        raise UnusableImageError('steering angle calculation failed')

    # Get the errors from the output values and add them to their corresponding accumulators
    errors = output_values[1:]
    for j in range(num_errors):
        accumulators[j] += errors[j] ** 2

    # Copy the image for use in the heat map section of the user interface
    heat_map_images = [numpy.copy(image) for _ in range(len(outer_road_lines))]

    # Create the dictionary of colors along with corresponding heat values
    heat_map_colors = {
        0.0: (0, 0, 128),
        0.25: (0, 0, 255),
        0.5: (0, 255, 0),
        0.75: (255, 255, 0),
        1.0: (255, 0, 0)
    }

    # Apply the heat map in place to the copied images, using a different inference engine for each
    for heat_map_image, inference_engine in zip(heat_map_images, inference_and_steering_wrapper.inference_engines):
        _apply_heat_map(heat_map_image, inference_engine.last_prediction_tensor, heat_map_colors, heat_map_opacity)

    # Calculate two points on the line of best fit
    line_parameters = inference_and_steering_wrapper.steering_engine.center_line_of_best_fit
    y_positions = (0, image.shape[0] - 1)
    x_positions = [
        int(round((y_position * line_parameters[1]) + line_parameters[0]))
        for y_position in y_positions
    ]

    # Transpose the list of Y positions followed by X positions and format it into a suitable list
    formatted_arguments = [
        value for position in zip(y_positions, x_positions)
        for value in position
    ]

    # Draw the line of best fit, catching IndexErrors on each iteration in case the line goes off the screen
    all_indices = line(*formatted_arguments)[:2]
    for indices in zip(*all_indices):
        try:
            image[indices] = 0
        except IndexError:
            pass

    # Remove the outliers from the center line positions
    center_line_positions_without_outliers = inference_and_steering_wrapper.steering_engine.remove_outliers(
        center_line_positions)

    # Display the center line in blue
    lines_and_colors = [(center_line_positions_without_outliers, [0, 0, 255])]
    # Display the outer lines in red and green (red if there is only one)
    red_and_green = (
        [255, 0, 0],
        [0, 255, 0]
    )
    for road_line_with_color in zip(outer_road_lines, red_and_green):
        lines_and_colors.append(road_line_with_color)

    # Add the relevant lines and points to the main image and scale it to double its original size
    _add_markers(image, marker_radius, lines_and_colors)
    image = imresize(image, 200, interp='nearest')

    # If there is only one heat map image, add a light gray rectangle of the same shape to the list
    if len(heat_map_images) == 1:
        empty_rectangle = numpy.copy(heat_map_images[0])
        empty_rectangle.fill(light_gray_color)
        heat_map_images.append(empty_rectangle)

    # Add a light gray border around the edge of all three images before tiling them
    for sub_image in heat_map_images + [image]:

        # Set the beginning and ending rows of both dimensions
        for x in [0, sub_image.shape[1] - 1]:
            sub_image[:, x] = light_gray_color
        for y in [0, sub_image.shape[0] - 1]:
            sub_image[y] = light_gray_color

    # Concatenate the two small images together and then concatenate them to the main image
    concatenated_heat_map_image = numpy.hstack(heat_map_images)
    tiled_image = numpy.vstack((image, concatenated_heat_map_image))

    # Return the steering angle and the image
    return tiled_image, output_values


# Add lines and points to the main image
def _add_markers(image, marker_radius, lines_and_colors):
    # For each of the two road lines
    for line_positions, color in lines_and_colors:

        # For each of the positions which include horizontal and vertical values
        for position in line_positions:
            # Calculate the four bounds of the marker to be placed
            bounds = [
                int(round(center + offset))
                for center in position
                for offset in (-marker_radius, marker_radius)
            ]

            # Create a square within the bounds
            image[bounds[0]:bounds[1], bounds[2]:bounds[3]] = color


# Display a translucent multi-colored heat map over an image (modifying it in place), given a tensor of
# predictions to base it on and a dictionary of colors with corresponding heat values to interpolate between
def _apply_heat_map(image, prediction_tensor, colors, heat_map_opacity):
    # Find the factor to calculate rectangular blocks in the image
    # that visually correspond to the positions in the prediction tensor
    heat_map_block_shape = [
        image_dimension / prediction_dimension
        for image_dimension, prediction_dimension in zip(
            image.shape, prediction_tensor.shape)
    ]

    # Iterate over both dimensions of the image
    for y_position, x_position in numpy.ndindex(prediction_tensor.shape):

        # Find the bounds of the corresponding heat map box in the image and slice out the box
        block_bounds = [
            int(round((dimension + offset) * block_dimension))
            for dimension, block_dimension in zip((
                y_position, x_position), heat_map_block_shape)
            for offset in (0, 1)
        ]
        block = image[block_bounds[0]:block_bounds[1], block_bounds[2]:block_bounds[3]]

        # Color calculated in the following loop
        interpolated_color = []

        # Temporary storage for the previous item used in every iteration of the loop
        previous_item = None

        # Compute the color from the dictionary by iterating over it and finding the first one that is greater than
        # the prediction tensor value corresponding to the image
        current_prediction = prediction_tensor[y_position, x_position]
        for heat_value, color in colors.iteritems():

            # If the first value is equal to zero, skip the first iteration because the previous value is not yet set
            if heat_value >= current_prediction and heat_value:

                # Find the proportion of the distance between the previous heat key and the current one
                # that the heat value of the current prediction cell is
                previous_heat_value = previous_item[0]
                interpolation_value = (current_prediction - previous_heat_value
                                       ) / (heat_value - previous_heat_value)

                # Loop over the previous and current color tuples and interpolate via a weighted average
                previous_color = previous_item[1]
                for previous, current in zip(previous_color, color):
                    weighted_average = (previous *
                                        (1 - interpolation_value)) + (
                                               current * interpolation_value)
                    interpolated_color.append(weighted_average)

                # Exit the loop because we have already completed the interpolation
                break

            # Remember the previous color for the next iteration
            previous_item = (heat_value, color)

        # Create the color block and give it the color calculated during interpolation
        color_block = numpy.zeros(block.shape)
        color_block[:, :] = interpolated_color

        # Color the block correspondingly, using the supplied opacity value and the calculated color
        block[:] = (block * (1 - heat_map_opacity)) + \
                   (color_block * heat_map_opacity)
