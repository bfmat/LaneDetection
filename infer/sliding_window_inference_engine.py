from __future__ import division

import numpy

from skimage.util import view_as_windows

# A system for efficiently running inference on sections of an image using sliding windows and calculating points down
# the center of the road based on the inference output
# Created by brendon-ai, September 2017


# A number used as a stride for the channel axis, ensuring that no window slicing occurs on that axis
MORE_THAN_CHANNELS = 6


# Main class, instantiated with configuration and trained model
class SlidingWindowInferenceEngine:
    # The Keras model used for inference on square sections of the image
    model = None

    # The side length of the sliding windows
    window_size = None

    # The size of gaps between positions where the sliding windows are used
    stride = None

    # Set-and-forget externally accessible storage for the prediction tensor of the most recently processed image
    last_prediction_tensor = None

    # Set global variables provided as arguments
    def __init__(self, model, slice_size, stride):

        # Set scalar variables
        self.model = model
        self.window_size = slice_size

        # If stride is a tuple, set the global variable to that tuple plus a term corresponding to the channel axis
        if isinstance(stride, tuple):
            self.stride = stride + (MORE_THAN_CHANNELS,)

        # If stride is a scalar, set the global variable to a three-element tuple with that value plus a channel term
        else:
            self.stride = (stride, stride, MORE_THAN_CHANNELS)

    # Given an image, compute a vector of positions describing the position of the line within each row
    def infer(self, image):

        # Slice up the image into windows
        image_slices = view_as_windows(
            image, (self.window_size, self.window_size, 3), self.stride)

        # Flatten the window array so that there is only one non-window dimension
        image_slices_flat = numpy.reshape(
            image_slices, (-1,) + image_slices.shape[-3:])

        # Run a prediction on all of the windows at once
        predictions = self.model.predict(image_slices_flat)

        # Reshape the predictions to have the same initial two dimensions as the original list of image slices
        predictions_row_arranged = numpy.reshape(
            predictions, image_slices.shape[:2])

        # Save the prediction tensor so external scripts can access it if desired
        self.last_prediction_tensor = predictions_row_arranged

        return predictions_row_arranged
