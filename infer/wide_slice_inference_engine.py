from skimage.util import view_as_windows


# A system for slicing an images and performing object detection inference on the slices
# Created by brendon-ai, September 2017


# Main class, instantiated with configuration and trained model
class WideSliceInferenceEngine:

    # Convolutional neural network model that accepts the entire wide image
    model = None

    # Dimensions of slices to take out of the main image
    slice_dimensions = None

    # Vertical distance between the wide slices the image is cut into
    vertical_stride = None

    # Set the model, slice height and vertical stride provided as an argument
    def __init__(self, model, slice_dimensions, vertical_stride):

        # Set global variables
        self.model = model
        self.slice_dimensions = slice_dimensions
        self.vertical_stride = vertical_stride

    # Given an image, compute a vector of positions describing the position of the line within each row
    def infer(self, image):

        # Split the image into wide slices
        image_slices = view_as_windows(image, self.slice_dimensions, self.vertical_stride)

        # A list that will contain the line positions for each row
        line_positions = []

        # Iterate over the slices
        for image_slice in image_slices:

            # Use the network to calculate the position of the object in the image
            lateral_position = self.model.predict(image_slice)

            # Add it to the list for return
            line_positions.append(lateral_position)

        return line_positions
