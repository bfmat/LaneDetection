import numpy


# A system for slicing an images and performing object detection inference on the slices
# Created by brendon-ai, September 2017


# Main class, instantiated with configuration and trained model
class WideSliceInferenceEngine:

    # Convolutional neural network model that accepts the entire wide image
    model = None

    # Dimensions of slices to take out of the main image
    slice_size = None

    # Vertical distance between the wide slices the image is cut into
    stride = None

    # Set the model, slice height and vertical stride provided as an argument
    def __init__(self, model, slice_size, stride):

        # Set global variables
        self.model = model
        self.slice_size = slice_size
        self.stride = stride

    # Given an image, compute a vector of positions describing the position of the line within each row
    def infer(self, image):

        # Split the image into wide slices
        image_slices = vertical_sliding_window(image, self.slice_size, self.stride)

        # Use the network to calculate the positions of the objects in each of the images
        network_outputs = self.model.predict(image_slices)[:, 0]

        # Convert the network's outputs to a list
        horizontal_positions = [position for position in network_outputs]

        # Calculate the vertical positions using the slice height
        vertical_positions = [(slice_index + 0.5) * self.slice_size for slice_index in range(len(image_slices))]

        # Compose a position tuple and add it to the list for return
        line_positions = [(horizontal_position, vertical_position)
                          for horizontal_position, vertical_position in zip(horizontal_positions, vertical_positions)]

        return line_positions


# Split an image vertically into slices given a stride
def vertical_sliding_window(image, slice_height, stride):

    # Get the height of the image
    image_height = image.shape[0]

    # List to add windows to
    window_list = []

    # Move across the image by the stride, starting at the left edge and stopping before going over the edge
    safe_end_position = image_height - slice_height
    for window_start in range(0, safe_end_position, stride):

        # Cut out a square window with the height of the image
        window_end = window_start + slice_height
        window = image[window_start:window_end]

        # Add it to the list
        window_list.append(window)

    # Convert it to a NumPy array before returning
    return numpy.array(window_list)
