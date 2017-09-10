from skimage.util import view_as_windows


# A system for running inference on sections of an image using sliding windows
# Created by brendon-ai, September 2017


# Main class, instantiated with configuration and trained model
class SlidingWindowInferenceEngine:

    # The Keras model used for inference on square sections of the image
    model = None

    # The side length of the sliding windows
    window_size = None

    # The size of gaps between positions where the sliding windows are used
    stride = None

    # Set global variables provided as arguments
    def __init__(self, model, window_size, stride):
        self.model = model
        self.window_size = window_size
        self.stride = stride

    # Given an image, compute a Boolean matrix containing the network's classifications of the corresponding window
    def infer(self, image):

        # Slice up the image into windows
        image_slices = view_as_windows(image, (3, self.window_size, self.window_size), self.stride)

        # A two-dimensional list that will contain the predictions corresponding to the windows
        classification_matrix = []

        # Loop over the windows and classify them, one row at a time
        for row in image_slices:

            # Create a one-dimensional list of the classifications for this row
            row_classifications = []

            # Loop over the second dimension
            for window in row:

                # Use the model to classify the image, and convert the result to a Boolean
                classification = bool(self.model.predict(window))

                # Append it to the list of classifications for this row
                row_classifications.append(classification)

            # Append this row's classifications to the matrix for the entire image
            classification_matrix.append(row_classifications)

        return classification_matrix
