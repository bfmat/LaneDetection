import cv2
import numpy as np
from skimage.util import view_as_windows

# Inference functionality that processes images and adds bounding boxes around stop signs
# Created by brendon-ai, April 2018

# Side length of the square windows to cut the images into
WINDOW_SIZE = 16
# Minimum area for a blob to be recognized
MIN_AREA = 16
# Number of pixels between the center and the edge of a bounding box, per size unit of the corresponding blob
BOX_SCALE = 0.7
# RGB color of the bounding boxes and labels around stop signs
BOX_COLOR = (255, 128, 0)


def box_stop_signs(model, search_image, draw_image):
    """Perform stop sign detection provided a model, an image to search for stop signs, and an image to draw bounding boxes on"""

    # Create parameters for the blob detector that will be used to find hot spots on the prediction map
    parameters = cv2.SimpleBlobDetector_Params()
    parameters.filterByCircularity = False
    parameters.filterByConvexity = False
    parameters.filterByInertia = False
    parameters.filterByColor = False
    parameters.minArea = MIN_AREA
    # Create the blob detector with the above parameters
    blob_detector = cv2.SimpleBlobDetector_create(parameters)

    # Slice up the image into windows
    image_slices = view_as_windows(
        search_image, (WINDOW_SIZE, WINDOW_SIZE, 3))
    # Flatten the window array so that there is only one non-window dimension
    image_slices_flat = np.reshape(
        image_slices, (-1,) + image_slices.shape[-3:])
    # Run a prediction on all of the windows at once
    predictions = model.predict(image_slices_flat)
    # Reshape the predictions to have the same initial two dimensions as the original list of image slices
    predictions_row_arranged = np.reshape(predictions, image_slices.shape[:2])
    # Convert the floating-point array to unsigned 8-bit integers, multiplying it by 255 to scale it into the range of 0 to 255, instead of 0 to 1
    predictions_integer = np.array(
        predictions_row_arranged * 255, dtype=np.uint8)

    # Find blobs in the heat map, which will be located around the location of the stop signs
    blob_key_points = blob_detector.detect(predictions_integer)
    # Convert the key points to tuple positions
    blob_positions, blob_sizes = zip(*((key_point.pt, key_point.size)
                                       for key_point in blob_key_points))
    # Transpose the shapes of the heat map and the image into (X, Y) format
    heat_map_shape, image_shape = [
        (array.shape[1], array.shape[0])
        for array in [predictions_integer, draw_image]
    ]
    # Reduce the image's effective size by the window size, to represent the areas on the edges that the windows are never centered at
    image_shape = tuple(shape_element - WINDOW_SIZE
                        for shape_element in image_shape)

    # Scale the blob positions to correspond to positions on the original image
    # Increase all values by half of the window size, so the reduced image size is inset into the available space
    blob_positions_scaled = [
        [
            (position_element / heat_map_dimension)
            * image_dimension + (WINDOW_SIZE // 2)
            for position_element, heat_map_dimension, image_dimension
            in zip(position, heat_map_shape, image_shape)
        ]
        for position in blob_positions
    ]

    # Draw each of the blobs as a box with a corresponding label on the output image
    for position, size in zip(blob_positions_scaled, blob_sizes):
        # Multiply the size of the blob by the scale factor to get the distance between the center and the edge of the box
        half_edge_length = size * BOX_SCALE
        # Get the positions of the top left and bottom right corners of the bounding box
        top_left = tuple(int(position_element - half_edge_length)
                         for position_element in position)
        bottom_right = tuple(int(position_element + half_edge_length)
                             for position_element in position)
        # Draw a black rectangle on the output image with these corners
        cv2.rectangle(
            img=draw_image,
            pt1=top_left,
            pt2=bottom_right,
            color=BOX_COLOR
        )

        # Offset the label horizontally by a bit more than half of the edge length, and bring it down vertically so it is roughly centered on the side of the image
        label_position = (
            int(round(position[0] + half_edge_length + 2)),
            int(round(position[1] + 4))
        )
        # Approximate the distance to the sign using a reciprocal function of the blob size
        distance_meters = int(round((1/size) * 200))
        # Draw a text label containing the distance
        cv2.putText(
            img=draw_image,
            text='{} m'.format(distance_meters),
            org=label_position,
            fontFace=2,
            fontScale=0.4,
            color=BOX_COLOR
        )
