from __future__ import division, print_function

import os
import numpy

from scipy.misc import imread, imresize
from ..infer.steering_engine import remove_outliers


# A collection of functions required for loading and processing the data for the visualizer
# including many of the elements that appear to be part of the interface but really are image
# modifications, such as line markers, center lines, and the network prediction heat map
# Created by brendon-ai, October 2017


# Load and process the image with the provided inference engines and steering engine
def process_images(image_folder, inference_engines, steering_engine, marker_radius):

    # Notify the user that we have started loading the images
    print('Loading images...')

    # List that we will add processed images to
    image_list = []

    # List of file names of images
    image_names = sorted(os.listdir(image_folder))

    # List of image metadata for display
    all_image_data = []

    # Loop over each of the images in the folder
    for image_name in image_names:

        # Print out the file name of the image
        print('Loaded {}'.format(image_name))

        # Load the image from disk, using its fully qualified path
        image_path = image_folder + '/' + image_name
        image = imread(image_path)

        # Process the image and add various markings to it, recording metadata returned for display purposes
        processed_image, image_data = _process_single_image(image, inference_engines, steering_engine, marker_radius)

        # Add the prepared image to the list
        image_list.append(processed_image)

        # Add the corresponding name, center position, error, and steering angle to the data list
        all_image_data.append((image_name,) + image_data)

    # Notify the user that loading is complete
    print('Loading complete!')

    # Return the images and their corresponding names
    return image_list, all_image_data


# Perform all necessary processing on a single image to prepare it for visualization
def _process_single_image(image, inference_engines, steering_engine, marker_radius):

    # List of points on the lines
    all_line_positions = []

    # With each of the provided engines
    for inference_engine in inference_engines:

        # Perform inference on the current image, adding the results to the list of points
        all_line_positions.append(inference_engine.infer(image))

    # Calculate a steering angle from the points
    values = steering_engine.compute_steering_angle(all_line_positions)

    # Set the steering angle and error to large negative values if None is returned
    if values is None:
        steering_angle = -5
        error = -5

    else:
        # Extract steering angle and error from the return values
        steering_angle, error = values

    # Remove the outliers from each of the lists and keep the outliers in a separate list
    all_line_positions_without_outliers = []
    all_line_positions_outliers_only = []
    for line_positions in all_line_positions:

        # Add the outlier-free points to one list
        line_positions_without_outliers = remove_outliers(line_positions, 40)
        all_line_positions_without_outliers.append(line_positions_without_outliers)

        # Find the points that are in the original list but not in the outlier-free list (that is, the outliers)
        all_positions_set = set(line_positions)
        outlier_free_set = set(line_positions_without_outliers)
        line_positions_outliers_only = list(all_positions_set - outlier_free_set)

        # Add it to the outlier-only list
        all_line_positions_outliers_only.append(line_positions_outliers_only)

    # Calculate the center of the road from the steering angle
    center_x = int(steering_engine.ideal_center_x - error)

    # Combine the two road lines with the lists of outliers, along with their corresponding colors
    lines_and_colors = [
        (all_line_positions_without_outliers[0], [0, 0, 255]),
        (all_line_positions_without_outliers[1], [0, 255, 0]),
        (all_line_positions_outliers_only[0], [255, 0, 0]),
        (all_line_positions_outliers_only[1], [255, 255, 0])
    ]

    # Copy the image twice for use in the heat map section of the user interface
    heat_map_images = [numpy.copy(image) for _ in range(2)]

    # Apply the heat map in place to the copied images, using a different inference engine for each
    for heat_map_image, inference_engine in zip(heat_map_images, inference_engines):
        _apply_heat_map(heat_map_image, inference_engine.last_prediction_tensor)

    # Downscale the heat map images to half of their original size
    downscaled_heat_map_images = [imresize(heat_map_image, 0.5) for heat_map_image in heat_map_images]

    # Add the relevant lines and points to the original image
    _add_markers(image, steering_engine, marker_radius, center_x, lines_and_colors)

    # Concatenate the two small images together and then concatenate them to the main image
    concatenated_heat_map_image = numpy.concatenate(downscaled_heat_map_images, axis=1)
    tiled_image = numpy.concatenate((image, concatenated_heat_map_image), axis=0)

    # Return relevant metadata about the image as well as the image itself
    return tiled_image, (center_x, error, steering_angle)


# Add lines and points to an image
def _add_markers(image, steering_engine, marker_radius, center_x, lines_and_colors):
    # Create a vertical blue line at the same X position as the predicted center of the road, if possible
    try:
        image[:, center_x] = [0, 0, 255]
    except:
        pass

    # Create a vertical black line at the predefined center of the image
    image[:, steering_engine.ideal_center_x] = 0

    # For each of the two road lines
    for line_positions, color in lines_and_colors:

        # For each of the positions which include horizontal and vertical values
        for position in line_positions:

            # Calculate the four bounds of the marker to be placed
            bounds = [int(round(center + offset)) for center in position for offset in (-marker_radius, marker_radius)]

            # Create a black square within the bounds
            image[bounds[2]:bounds[3], bounds[0]:bounds[1]] = color


# Display a translucent red heat map over an image (modifying it in place), given a tensor of predictions to base it on
def _apply_heat_map(image, prediction_tensor):

    # Find the factor to calculate rectangular blocks in the image
    # that visually correspond to the positions in the prediction tensor
    heat_map_block_shape = [image_dimension / prediction_dimension
                            for image_dimension, prediction_dimension in zip(image.shape, prediction_tensor.shape)]

    # Iterate over both dimensions of the image
    for y, x in numpy.ndindex(prediction_tensor.shape):

        # Find the bounds of the corresponding heat map box in the image and slice out the box
        block_bounds = [
            int(round((dimension + offset) * block_dimension))
            for dimension, block_dimension in zip((y, x), heat_map_block_shape)
            for offset in (0, 1)
        ]
        block = image[block_bounds[0]:block_bounds[1], block_bounds[2]:block_bounds[3]]

        # Color shift the block to red, with intensity of the red
        # proportional to the prediction corresponding to the block
        red_weight = prediction_tensor[y, x]
        red_block = numpy.zeros(block.shape)
        red_block[:, :, 0] = 255 * red_weight
        block[:] = block * (1 - red_weight) + red_block
